import asyncio

from langchain_openai import ChatOpenAI
from browser_use.agent.service import CopyCatStep
from browser_use import Browser, BrowserConfig, Agent
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

async def run_local_browser_use():
    while True:
        task = input("Enter prompt: ")
        
        browser = Browser(
            config=BrowserConfig(
                chrome_instance_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
                extra_chromium_args=[
                    "--disable-session-crashed-bubble",
                    "--hide-crash-restore-bubble",
                ],
            )
        )
        
        agent = Agent(
            copycat_step=CopyCatStep(
                description=task,
            ),
            llm=llm,
            browser=browser,
        )
        
        await agent.run()
    
    
    
if __name__ == "__main__":
    asyncio.run(run_local_browser_use())
    