# Smolvlm-256m-instruct

This Plugin is a Vision model for Aeon

---
## Plugin Config
```yaml
aeon_plugin:
  plugin_name: smolvlm-256m-instruct
  type: image-text
  model_path: ./model/
  command: /view
  parameters: <PATH> <PROMPT>
  desc: Used to interpret images
```

## Plugin Structure

The plugin is organized into a single directory that contains its configuration and main script.

```
plugins/
└── aeon/
    ├── model/
    ├── config.yml
    ├── requirements.txt
    └── main.py

```


