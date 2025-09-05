# SmolVLM-256m-instruct

Allows the AI to "see" and interpret images. By linking visual data with language, it can describe scenes, answer visual queries, and identify objects within pictures.

Original model: [HuggingFaceTB/SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)

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


