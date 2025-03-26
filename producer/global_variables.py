from pathlib import Path


class ProducerGlobalVariables:
    CODE_ROOT = Path(__file__).resolve().parent
    PROJECT_ROOT = CODE_ROOT.parent