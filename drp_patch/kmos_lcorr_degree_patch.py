"""EMPOWER patch for KMOS DRP 4.5.11: polynomial degree of the OH-line wavelength correction from the environment.

kmos/kmo_priv_lcorr.c, kmo_lcorr_crosscorrelate_spectra, fits Δλ(λ) per exposure and IFU with a hard-coded 2nd-order
polynomial (`cpl_size degree[1] = {2};`). After this patch the degree is read from KMOS_LCORR_DEGREE (0–4); unset or empty
keeps 2, i.e. the original behaviour. The recipe log reports the degree once per run.

    python3 kmos_lcorr_degree_patch.py <kmos-4.5.11 source dir>      # edits kmos/kmo_priv_lcorr.c in place, once
"""
import pathlib
import sys

INCLUDE_OLD = "#include <math.h>\n#include <stdio.h>\n"
INCLUDE_NEW = "#include <math.h>\n#include <stdio.h>\n#include <stdlib.h>\n"
DEGREE_OLD = "        cpl_size degree[1] = {2};\n"
DEGREE_NEW = DEGREE_OLD + """\
        /* EMPOWER patch: degree from KMOS_LCORR_DEGREE (0-4); unset or empty = 2 (original) */
        {
            static int reported = 0;
            const char *env_deg = getenv("KMOS_LCORR_DEGREE");
            if (env_deg != NULL && *env_deg != '\\0') {
                char *end = NULL;
                long d = strtol(env_deg, &end, 10);
                if (end != env_deg && *end == '\\0' && d >= 0 && d <= 4) {
                    degree[0] = (cpl_size)d;
                } else if (!reported) {
                    cpl_msg_warning(__func__, "KMOS_LCORR_DEGREE=%s ignored (allowed 0-4)", env_deg);
                }
            }
            if (!reported) {
                cpl_msg_info(__func__, "OH lambda correction: polynomial degree %lld (KMOS_LCORR_DEGREE %s)",
                             (long long)degree[0], env_deg != NULL ? env_deg : "unset");
                reported = 1;
            }
        }
"""

src = pathlib.Path(sys.argv[1]) / "kmos" / "kmo_priv_lcorr.c"
text = src.read_text()
if "KMOS_LCORR_DEGREE" in text:
    sys.exit(f"{src}: already patched")
assert text.count(INCLUDE_OLD) == 1 and text.count(DEGREE_OLD) == 1, "unexpected source, patch not applied"
src.write_text(text.replace(INCLUDE_OLD, INCLUDE_NEW).replace(DEGREE_OLD, DEGREE_NEW))
print(f"{src}: patched")
