from jinja2 import Environment, FileSystemLoader, meta
import os
class PromptManager:
    def __init__(self, prompt_dir: str):
        self.env = Environment(loader=FileSystemLoader(prompt_dir))
        self.prompt_dir = prompt_dir

    def get(self, key: str, **kwargs):
        template = self.env.get_template(f"{key}.txt")
        return template.render(**kwargs)

    def list_prompts(self) -> list[str]:
        return [
            f[:-4] for f in os.listdir(self.prompt_dir)
            if f.endswith(".txt")
        ]

    def get_params(self, key: str) -> set[str]:
        """提取模板需要的参数名"""
        template_src = self.env.loader.get_source(self.env, f"{key}.txt")[0]
        parsed = self.env.parse(template_src)
        return meta.find_undeclared_variables(parsed)

if __name__ == "__main__":
    pm = PromptManager("./prompts")

    print(pm.list_prompts())
    # ['greeting', 'farewell']

    print(pm.get_params("greeting"))
    # {'name', 'place'}

    print(pm.get_params("farewell"))
    # {'user', 'date'}
