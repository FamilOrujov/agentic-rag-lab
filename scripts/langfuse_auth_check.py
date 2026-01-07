from langfuse import get_client

lf = get_client()
print("auth_check:", lf.auth_check())
