/* vcd_toggles: per-signal toggle counts and high times from a VCD stream (a file, or the FIFO that flow/gls.sh gives
 * the simulator), so that a whole measurement window is counted without storing the waveform.
 *
 * Counting rules of OpenSTA's read_power_activities -vcd (2.6.0): the initial value does not count; a value change
 * counts one transition, or 0.5 when x or z is involved; the high time accumulates while the value is '1'; the window
 * ends at the last timestamp of the file.
 *
 * Output: the header lines #timescale_s, #time_min, #time_max, #vars, #aliases, then one line per bit,
 * "<scope/path>\t<transitions>\t<high_time>[\t<bit>]" (a bus bit carries its index in a fourth column, because the
 * name may be an escaped identifier with brackets of its own). An id with several aliases (one net seen from several
 * scopes) is written once per alias.
 *
 * Build: cc -O2 -o flow/tools/vcd_toggles flow/tools/vcd_toggles.c   (flow/gls.sh builds it when the source changes)
 * Usage: vcd_toggles [-o out.tsv] [--begin T] [--end T] [vcd | -]
 *   --begin/--end  count only within [T_begin, T_end] (VCD time units), as read_power_activities -begin/-end do; the
 *                  reference flow uses neither (its testbench starts the waveform at the window start).
 */
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {                    /* one $var line */
  char *name;                       /* scope path and variable name */
  int width, msb, lsb;
  int has_range;                    /* 0: none, 1: [msb:lsb], 2: [bit] */
} Alias;

typedef struct {                    /* one VCD id: width bits, one or more aliases */
  uint64_t key;
  int width, n_alias;
  Alias *alias;
  char *prev;                       /* per bit: the last value ... */
  long long *prev_t;                /* ... and when it was set (-1: no value yet) */
  double *tc;                       /* per bit: transitions */
  unsigned long long *high;         /* per bit: time at '1' */
} Sig;

static Sig *sigs;
static size_t nsig, capsig;
static uint32_t *htab;              /* open addressing: htab[h] = signal index + 1 */
static size_t hcap;
static long long tmin = -1, tmax = 0, tcur = 0;
static double tscale = 1e-9;
static long long wbeg = -1, wend = -1;   /* the optional window [wbeg, wend] in VCD time units; -1 = unbounded */
static char scope[64][256];
static int depth;

/* VCD ids are 1..8 printable characters: packed into one 64-bit key with the length in the top byte */
static uint64_t idkey(const char *s, size_t n) {
  uint64_t k = 0;
  if (n > 8) {
    fprintf(stderr, "vcd_toggles: id too long: %.*s\n", (int)n, s);
    exit(1);
  }
  memcpy(&k, s, n);
  return k ^ ((uint64_t)n << 56);
}

static uint64_t hash(uint64_t k) {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33;
  return k;
}

static void grow_table(void) {
  size_t cap = hcap ? hcap * 2 : 1 << 16;
  uint32_t *table = calloc(cap, sizeof *table);
  for (size_t i = 0; i < nsig; i++) {
    size_t h = hash(sigs[i].key) & (cap - 1);
    while (table[h]) h = (h + 1) & (cap - 1);
    table[h] = (uint32_t)i + 1;
  }
  free(htab);
  htab = table;
  hcap = cap;
}

static Sig *find(uint64_t key, int create) {
  if (hcap == 0 || nsig * 2 >= hcap) grow_table();
  size_t h = hash(key) & (hcap - 1);
  while (htab[h]) {
    Sig *s = &sigs[htab[h] - 1];
    if (s->key == key) return s;
    h = (h + 1) & (hcap - 1);
  }
  if (!create) return NULL;
  if (nsig == capsig) {
    capsig = capsig ? capsig * 2 : 4096;
    sigs = realloc(sigs, capsig * sizeof *sigs);
    grow_table();
    h = hash(key) & (hcap - 1);
    while (htab[h]) h = (h + 1) & (hcap - 1);
  }
  Sig *s = &sigs[nsig];
  memset(s, 0, sizeof *s);
  s->key = key;
  htab[h] = (uint32_t)nsig + 1;
  nsig++;
  return s;
}

static void alloc_bits(Sig *s, int width) {
  if (s->width) return;
  s->width = width;
  s->prev = calloc(width, 1);
  s->prev_t = malloc(width * sizeof(long long));
  for (int i = 0; i < width; i++) s->prev_t[i] = -1;
  s->tc = calloc(width, sizeof(double));
  s->high = calloc(width, sizeof(unsigned long long));
}

/* a new value v of bit `bit` at time t */
static void update(Sig *s, int bit, long long t, char v) {
  if (v == 'X') v = 'x';
  if (v == 'Z') v = 'z';
  int in_window = (wbeg < 0 || t >= wbeg) && (wend < 0 || t <= wend);
  if (s->prev_t[bit] >= 0 && in_window) {
    char p = s->prev[bit];
    long long start = s->prev_t[bit];
    if (wbeg >= 0 && start < wbeg) start = wbeg;         /* clip the high interval to the window */
    if (p == '1' && t > start) s->high[bit] += (unsigned long long)(t - start);
    if (v != p) s->tc[bit] += (v == 'x' || v == 'z' || p == 'x' || p == 'z') ? 0.5 : 1.0;
  }
  if (wend < 0 || t <= wend) {
    s->prev[bit] = v;
    s->prev_t[bit] = t;
  }
}

/* "1ps", "1 ns", "10ns" -> seconds */
static double parse_timescale(const char *body) {
  double n = atof(body);
  const char *u = body;
  while (*u && (isdigit((unsigned char)*u) || isspace((unsigned char)*u) || *u == '.')) u++;
  double unit = 1;
  if (!strncmp(u, "ms", 2)) unit = 1e-3;
  else if (!strncmp(u, "us", 2)) unit = 1e-6;
  else if (!strncmp(u, "ns", 2)) unit = 1e-9;
  else if (!strncmp(u, "ps", 2)) unit = 1e-12;
  else if (!strncmp(u, "fs", 2)) unit = 1e-15;
  return (n > 0 ? n : 1) * unit;
}

/* $var wire 24 , y1 [23:0] $end   |   $var wire 1 . \x[9] $end   (escaped names may contain brackets) */
static void parse_var(const char *line, size_t *nvar, size_t *nalias) {
  char type[32], id[16], name[512], range[64] = "";
  int width = 0;
  int k = sscanf(line + 4, " %31s %d %15s %511s %63s", type, &width, id, name, range);
  if (k < 4 || width <= 0) return;
  if (k == 5 && !strcmp(range, "$end")) range[0] = 0;
  Sig *s = find(idkey(id, strlen(id)), 1);
  alloc_bits(s, width);
  char path[4096] = "";
  for (int d = 0; d < depth && d < 63; d++) {
    strcat(path, scope[d]);
    strcat(path, "/");
  }
  strcat(path, name);
  s->alias = realloc(s->alias, (s->n_alias + 1) * sizeof(Alias));
  Alias *a = &s->alias[s->n_alias++];
  a->name = strdup(path);
  a->width = width;
  a->has_range = 0;
  (*nvar)++;
  if (range[0] == '[') {
    int m, l;
    if (sscanf(range, "[%d:%d]", &m, &l) == 2) {
      a->has_range = 1;
      a->msb = m;
      a->lsb = l;
    } else if (sscanf(range, "[%d]", &m) == 1) {
      a->has_range = 2;
      a->msb = a->lsb = m;
    }
  }
  if (s->n_alias > 1) (*nalias)++;
}

int main(int argc, char **argv) {
  const char *out = NULL, *in = NULL;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
    else if (!strcmp(argv[i], "--begin") && i + 1 < argc) wbeg = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--end") && i + 1 < argc) wend = atoll(argv[++i]);
    else in = argv[i];
  }
  FILE *f = (!in || !strcmp(in, "-")) ? stdin : fopen(in, "r");
  if (!f) {
    perror(in);
    return 1;
  }
  static char buf[1 << 22];
  setvbuf(f, buf, _IOFBF, sizeof buf);

  char *line = NULL;
  size_t cap = 0, nvar = 0, nalias = 0;
  ssize_t len;
  int header = 1;
  while ((len = getline(&line, &cap, f)) > 0) {
    while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = 0;
    if (len == 0) continue;
    if (header) {                    /* the declarations: scopes, variables, timescale */
      if (!strncmp(line, "$scope", 6)) {
        char type[64], name[256];
        if (sscanf(line + 6, " %63s %255s", type, name) == 2) {
          if (depth < 63) strcpy(scope[depth], name);
          depth++;
        }
      } else if (!strncmp(line, "$upscope", 8)) {
        if (depth > 0) depth--;
      } else if (!strncmp(line, "$var", 4)) {
        parse_var(line, &nvar, &nalias);
      } else if (!strncmp(line, "$timescale", 10)) {     /* on the same line or the next one */
        char body[128] = "";
        const char *p = line + 10;
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p && getline(&line, &cap, f) > 0)
          for (p = line; *p && isspace((unsigned char)*p); p++) {}
        strncpy(body, p, 127);
        char *e = strstr(body, "$end");
        if (e) *e = 0;
        tscale = parse_timescale(body);
      } else if (!strncmp(line, "$enddefinitions", 15)) {
        header = 0;
      }
      continue;
    }
    char c = line[0];
    if (c == '#') {                  /* a timestamp */
      tcur = atoll(line + 1);
      if (tmin < 0) tmin = tcur;
      if (tcur > tmax) tmax = tcur;
    } else if (c == '0' || c == '1' || c == 'x' || c == 'z' || c == 'X' || c == 'Z') {   /* a scalar change */
      Sig *s = find(idkey(line + 1, len - 1), 0);
      if (s) update(s, 0, tcur, c);
    } else if (c == 'b' || c == 'B') {                   /* a vector change, left-extended to the width */
      char *sp = strchr(line, ' ');
      if (!sp) continue;
      *sp = 0;
      const char *val = line + 1, *id = sp + 1;
      size_t vl = strlen(val);
      Sig *s = find(idkey(id, strlen(id)), 0);
      if (!s) continue;
      int w = s->width;
      char ext = (val[0] == 'x' || val[0] == 'X') ? 'x' : (val[0] == 'z' || val[0] == 'Z') ? 'z' : '0';
      for (int b = 0; b < w; b++) {                      /* b = 0 is the most significant bit (first character) */
        long long pos = (long long)vl - w + b;
        update(s, b, tcur, pos >= 0 ? val[pos] : ext);
      }
    }
    /* $dumpvars, $end, $dumpall, $comment and real values: ignored */
  }

  if (wbeg >= 0 && (tmin < 0 || tmin < wbeg)) tmin = wbeg;   /* report the window as the duration */
  if (wend >= 0 && tmax > wend) tmax = wend;
  FILE *o = out ? fopen(out, "w") : stdout;
  if (!o) {
    perror(out);
    return 1;
  }
  fprintf(o, "#timescale_s\t%g\n#time_min\t%lld\n#time_max\t%lld\n#vars\t%zu\n#aliases\t%zu\n",
          tscale, tmin < 0 ? 0 : tmin, tmax, nvar, nalias);
  for (size_t i = 0; i < nsig; i++) {
    Sig *s = &sigs[i];
    for (int a = 0; a < s->n_alias; a++) {
      Alias *al = &s->alias[a];
      for (int b = 0; b < s->width; b++) {
        unsigned long long high = s->high[b];
        if (s->prev_t[b] >= 0 && s->prev[b] == '1' && tmax > s->prev_t[b])   /* still '1' at the end */
          high += (unsigned long long)(tmax - s->prev_t[b]);
        if (s->width == 1 && al->has_range == 0) {
          fprintf(o, "%s\t%.1f\t%llu\n", al->name, s->tc[b], high);
        } else {
          int bit = al->has_range ? (al->msb >= al->lsb ? al->msb - b : al->msb + b) : (s->width - 1 - b);
          fprintf(o, "%s\t%.1f\t%llu\t%d\n", al->name, s->tc[b], high, bit);
        }
      }
    }
  }
  if (o != stdout) fclose(o);
  fprintf(stderr, "vcd_toggles: %zu ids, %zu vars (%zu aliases), time %lld..%lld x %g s\n",
          nsig, nvar, nalias, tmin, tmax, tscale);
  return 0;
}
