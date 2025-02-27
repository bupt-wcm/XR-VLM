from libcore.libs.registry import model_register


def build_model(model_config, cls_num):
    return model_register.get(model_config.NAME)(model_config, cls_num)
