all = (
    'get_class_label'
)


def get_class_label(path: str) -> str:
    file_name = path.rpartition('/')[2]
    return file_name[5]