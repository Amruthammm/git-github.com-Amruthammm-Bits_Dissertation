# import pydantic

# # Print the version of pydantic
# print(f"pydantic version: {pydantic.VERSION}")

# Replace the content of langchain_core/utils/pydantic.py with the following:

# Replace the content of langchain_core/utils/pydantic.py with the following:

# from pydantic.version import VERSION as PYDANTIC_VERSION

# def get_pydantic_major_version():
#     try:
#         version_str = PYDANTIC_VERSION.split(".")[0]
#         return int(version_str)
#     except Exception as e:
#         print(f"Error getting pydantic version: {e}")
#         return 0

# PYDANTIC_MAJOR_VERSION = get_pydantic_major_version()

import pydantic

print(pydantic.VERSION)



