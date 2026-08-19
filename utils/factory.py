def get_model(model_name, args):
    name = model_name.lower()
    if name in ('prism', 'tosca'):
        # 'tosca' kept as a back-compat alias for any old exp config or
        # resumed run still using the pre-rename model_name (Prism's Learner
        # uses the TOSCA adapter internally; see models/prism.py).
        from models.prism import Learner
    else:
        raise NotImplementedError("Unknown model: {}".format(model_name))
    return Learner(args)
