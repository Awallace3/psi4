"""Admission and original-monomer wiring; source solves/readout stay native."""


def validate_request(core, *, external_potentials=None):
    if not core.get_option("FISAPT", "FISAPT_FSAPT_IND_XC_RESPONSE"):
        return False
    if ("BRIANQC_ENABLE" in core.get_global_option_list()
            and core.get_global_option("BRIANQC_ENABLE")):
        raise ValueError("CPKS find requires CPU monomers")
    if external_potentials:
        raise NotImplementedError("CPKS find does not admit external potentials")
    if not core.get_option("FISAPT", "FISAPT_FSAPT_IND_RESPONSE"):
        raise ValueError("FISAPT_FSAPT_IND_XC_RESPONSE requires FISAPT_FSAPT_IND_RESPONSE=true")
    if (core.get_option("SAPT", "SAPT_DFT_DO_FSAPT") != "FISAPT"
            or core.get_option("SAPT", "SAPT_DFT_INDUCTION_TYPE") != "CPKS"):
        raise NotImplementedError("CPKS find requires native FISAPT and canonical CPKS")
    if (core.get_option("SAPT", "SAPT_DFT_FUNCTIONAL").upper() not in ("HF", "SVWN", "PBE")
            or core.get_option("SAPT", "SAPT_DFT_DO_DDFT")):
        raise NotImplementedError("CPKS find admits HF/SVWN/PBE without deltaDFT redistribution")
    if (core.get_option("SAPT", "SAPT_DFT_GRAC_COMPUTE") != "NONE"
            or core.get_option("SAPT", "SAPT_DFT_GRAC_SHIFT_A") != 0.
            or core.get_option("SAPT", "SAPT_DFT_GRAC_SHIFT_B") != 0.
            or core.get_option("SCF", "LEVEL_SHIFT") != 0.):
        raise NotImplementedError("CPKS find does not admit shifted references")
    if (core.get_option("FISAPT", "FISAPT_LINK_ASSIGNMENT") != "C"
            or core.get_option("FISAPT", "SSAPT0_SCALE")
            or core.get_option("FISAPT", "FISAPT_EXCH_PARPERP")
            or core.get_option("SAPT", "DO_IND_EXCH_SINF")):
        raise NotImplementedError("CPKS find requires C assignment and unscaled S2 readout")
    for name in ("PCM", "PE", "DDX"):
        if name in core.get_global_option_list() and core.get_option("SCF", name):
            raise NotImplementedError("CPKS find does not admit external response")
    return True


def install(core, fisapt, wfn_a, wfn_b):
    try:
        if validate_request(core):
            if not hasattr(fisapt, "set_cpks_monomers"):
                raise RuntimeError("CPKS find requires a matching native build")
            fisapt.set_cpks_monomers(wfn_a, wfn_b)
    finally:
        core.prepare_options_for_module("FISAPT")
