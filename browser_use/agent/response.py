import logging

from typing import Callable
from pydantic import BaseModel
from browser_use.agent.views import AgentOutput

logger = logging.getLogger(__name__)

class StepResponse(BaseModel):
    eval: str
    next_goal: str
    actions: list[str]

def log_response(
    response: AgentOutput, 
    on_log_step_response: Callable[[StepResponse], None] | None = None
) -> None:
	"""Utility function to log the model's response."""

	if 'Success' in response.current_state.evaluation_previous_goal:
		emoji = '😻'
	elif 'Failed' in response.current_state.evaluation_previous_goal:
		emoji = '😿'
	else:
		emoji = '😿'
  
	eval = f'{emoji} {response.current_state.evaluation_previous_goal}'
	memory = f'🧠 Memory: {response.current_state.memory}'
	next_goal = f'🐈 Next Goal: {response.current_state.next_goal}'
	actions = []
 
	for i, action in enumerate(response.action):
		if i == len(response.action) - 1:
			action = f'🐱  Result: {action.model_dump_json(exclude_unset=True, indent=2)}'
			actions.append(action)
		else:
			action = f'🐱  Action: {action.model_dump_json(exclude_unset=True, indent=2)}'
			actions.append(action)

	logger.info(eval)
	logger.info(memory)
	logger.info(next_goal)
 
	for action in actions:
		logger.info(action)
  
	if on_log_step_response:
		step_response = StepResponse(
			eval=eval, 
			next_goal=next_goal, 
			actions=actions
		)
		on_log_step_response(step_response)