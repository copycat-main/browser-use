import posthog

from posthog.ai.langchain import CallbackHandler

posthog.project_api_key = "phc_KqAFA7J95snGx0kEDbXhGri2oh2BtLOCs8Ell3TTK3d"
posthog.host = "https://us.i.posthog.com"

# def get_callback_handler() -> CallbackHandler:
#     return CallbackHandler(
#         client=posthog,
#         privacy_mode=False
#     )