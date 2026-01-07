from langfuse import get_client

lf = get_client()

with lf.start_as_current_observation(as_type="span", name="smoke-test") as span:
    span.update_trace(input={"ping": "hello"})
    span.update_trace(output={"pong": "world"})


lf.flush()
print("smoke trace sent")

