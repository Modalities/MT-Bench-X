from pydantic import BaseModel, FilePath


class EvalSet(BaseModel):
    question_file: FilePath
    answer_file: FilePath
