from pathlib import Path


class WorkerGlobalVariables:
    CODE_ROOT = Path(__file__).resolve().parent
    PROJECT_ROOT = CODE_ROOT.parent