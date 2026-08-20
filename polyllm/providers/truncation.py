#: Emitted as a ("notice", ...) chunk when generation stopped at the token cap
#: rather than finishing. Worth surfacing: thinking models spend the same budget
#: on reasoning, so a low max_tokens can swallow the answer entirely.
TRUNCATED = "Response was cut off at the max tokens limit."
