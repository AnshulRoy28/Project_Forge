"""State machine for project lifecycle."""

from enum import Enum


class State(str, Enum):
    """Project states in the pipeline."""
    
    INIT = "INIT"
    SCOPING = "SCOPING"
    SPEC_CONFIRMED = "SPEC_CONFIRMED"
    DATA_REQUIRED = "DATA_REQUIRED"
    DATA_VALIDATED = "DATA_VALIDATED"
    ENV_BUILDING = "ENV_BUILDING"
    ENV_READY = "ENV_READY"
    MOCK_RUNNING = "MOCK_RUNNING"
    MOCK_PASSED = "MOCK_PASSED"
    TRAINING = "TRAINING"
    TRAINING_COMPLETE = "TRAINING_COMPLETE"
    INFERENCE_READY = "INFERENCE_READY"
    DONE = "DONE"
    
    def can_transition_to(self, next_state: "State") -> bool:
        """Check if transition to next_state is valid."""
        valid_transitions = {
            State.INIT: [State.SCOPING],
            State.SCOPING: [State.SCOPING, State.SPEC_CONFIRMED],
            State.SPEC_CONFIRMED: [State.DATA_REQUIRED],
            State.DATA_REQUIRED: [State.DATA_VALIDATED],
            State.DATA_VALIDATED: [State.ENV_BUILDING],
            State.ENV_BUILDING: [State.ENV_READY],
            State.ENV_READY: [State.MOCK_RUNNING],
            State.MOCK_RUNNING: [State.MOCK_PASSED, State.MOCK_RUNNING],  # Can retry
            State.MOCK_PASSED: [State.TRAINING],
            State.TRAINING: [State.TRAINING_COMPLETE],
            State.TRAINING_COMPLETE: [State.INFERENCE_READY],
            State.INFERENCE_READY: [State.DONE],
            State.DONE: [],
        }
        
        return next_state in valid_transitions.get(self, [])
    
    def get_next_action(self) -> str:
        """Get human-readable next action for this state."""
        actions = {
            State.INIT: "Run 'nnb start' to begin",
            State.SCOPING: "Continue conversation or confirm specification",
            State.SPEC_CONFIRMED: "Review data requirements",
            State.DATA_REQUIRED: "Run 'nnb data validate --path <data-path>'",
            State.DATA_VALIDATED: "Run 'nnb env build'",
            State.ENV_BUILDING: "Wait for container build to complete",
            State.ENV_READY: "Run 'nnb mock-run'",
            State.MOCK_RUNNING: "Wait for mock run to complete",
            State.MOCK_PASSED: "Run 'nnb train'",
            State.TRAINING: "Wait for training or run 'nnb attach'",
            State.TRAINING_COMPLETE: "Review training results",
            State.INFERENCE_READY: "Test inference with 'nnb inference test'",
            State.DONE: "Project complete! 🎉",
        }
        
        return actions.get(self, "Unknown state")


class InvalidStateTransitionError(Exception):
    """Raised when attempting an invalid state transition."""
    
    def __init__(self, current: State, attempted: State):
        self.current = current
        self.attempted = attempted
        super().__init__(
            f"Cannot transition from {current.value} to {attempted.value}"
        )
