import yaml
config = yaml.safe_load(open("src/config.yaml"))
globals().update(config)
print(globals)