"""
Graph Builder for LangGraph SMART-LLM Pipeline using Claude API.
Drop-in replacement for graph_builder.py but with Claude instead of OpenAI.
"""

from langgraph.graph import StateGraph, END
from scripts.langgraph_pipeline.state.schema import PipelineState
from scripts.langgraph_pipeline.nodes.decomposition_node_claude import TaskDecompositionAgentClaude
from scripts.langgraph_pipeline.nodes.allocation_node_claude import RobotAllocationAgentClaude
from scripts.langgraph_pipeline.nodes.code_generation_node_claude import CodeGenerationAgentClaude
from scripts.langgraph_pipeline.edges.validators import validate_decomposition, validate_allocation, validate_code
from scripts.langgraph_pipeline.edges.routing_logic import route_next_agent


class LangGraphSMARTLLMClaude:
    """
    Main LangGraph-based SMART-LLM pipeline orchestrator using Claude API.
    Drop-in replacement for the OpenAI version.
    """
    
    def __init__(self, api_key, decomp_model="claude-sonnet-4-20250514", 
                 alloc_model="claude-sonnet-4-20250514", codegen_model="claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.decomp_model = decomp_model
        self.alloc_model = alloc_model
        self.codegen_model = codegen_model
        
        # Initialize agents
        self.decomposition_agent = TaskDecompositionAgentClaude(decomp_model, api_key)
        self.allocation_agent = RobotAllocationAgentClaude(alloc_model, api_key)
        self.code_generation_agent = CodeGenerationAgentClaude(codegen_model, api_key)
        
        # Build workflow
        self.workflow = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Create workflow
        workflow = StateGraph(PipelineState)
        
        # Add nodes
        workflow.add_node("decomposition", self.decomposition_agent)
        workflow.add_node("allocation", self.allocation_agent)
        workflow.add_node("code_generation", self.code_generation_agent)
        workflow.add_node("validate_decomposition", validate_decomposition)
        workflow.add_node("validate_allocation", validate_allocation)
        workflow.add_node("validate_code", validate_code)
        
        # Set entry point
        workflow.set_entry_point("decomposition")
        
        # Add conditional edges with Claude-specific routing logic
        workflow.add_conditional_edges(
            "decomposition",
            self.get_claude_next_node,
            {
                "validate_decomposition": "validate_decomposition",
                "failed": END,
                "complete": END
            }
        )
        
        workflow.add_conditional_edges(
            "validate_decomposition", 
            self.get_claude_next_node,
            {
                "allocation": "allocation",
                "decomposition": "decomposition",
                "failed": END,
                "complete": END
            }
        )
        
        workflow.add_conditional_edges(
            "allocation",
            self.get_claude_next_node,
            {
                "validate_allocation": "validate_allocation", 
                "failed": END,
                "complete": END
            }
        )
        
        workflow.add_conditional_edges(
            "validate_allocation",
            self.get_claude_next_node,
            {
                "code_generation": "code_generation",
                "allocation": "allocation",
                "decomposition": "decomposition",
                "failed": END,
                "complete": END
            }
        )
        
        workflow.add_conditional_edges(
            "code_generation",
            self.get_claude_next_node,
            {
                "validate_code": "validate_code",
                "failed": END,
                "complete": END
            }
        )
        
        workflow.add_conditional_edges(
            "validate_code",
            self.get_claude_next_node,
            {
                "complete": END,
                "code_generation": "code_generation",
                "allocation": "allocation", 
                "decomposition": "decomposition",
                "failed": END
            }
        )
        
        return workflow.compile()
    
    def get_claude_next_node(self, state: PipelineState) -> str:
        """
        Claude-specific routing logic that maps to Claude node names.
        """
        next_agent = route_next_agent(state)
        
        # Map agent types to Claude node names
        claude_node_mapping = {
            "decomposition": "decomposition",
            "allocation": "allocation", 
            "code_generation": "code_generation",
            "complete": "END",
            "failed": "END"
        }
        
        # Special handling for validation - route based on most recent completion
        if next_agent == "validation":
            validation_results = state.get('validation_results', {})
            
            # Check what needs validation (most recent first)
            if 'generated_code' in state and state['generated_code'] and not validation_results.get('code'):
                return "validate_code"
            elif 'allocation_plan' in state and state['allocation_plan'] and not validation_results.get('allocation'):
                return "validate_allocation"  
            elif 'decomposed_plan' in state and state['decomposed_plan'] and not validation_results.get('decomposition'):
                return "validate_decomposition"
            
            # If all validations are done, this shouldn't happen, but fallback to decomposition
            return "validate_decomposition"
        
        # For conditional edges, return the key that maps to the actual node
        if next_agent in ["complete", "failed"]:
            print(f"🔄 Claude Routing: {state.get('current_stage', 'unknown')} → {next_agent} (terminal)")
            return next_agent  # Return "complete" or "failed", not "END"
        
        node_name = claude_node_mapping.get(next_agent, "END")
        print(f"🔄 Claude Routing: {state.get('current_stage', 'unknown')} → {next_agent} (node: {node_name})")
        return node_name
    
    def process_single_task(self, task, available_robots, objects, ground_truth, trans, max_trans, 
                           floor_plan, examples):
        """Process a single task through the LangGraph pipeline."""
        
        print(f"\n📋 LangGraph Processing: {task}")
        print("=" * 60)
        
        # Initialize state with all required fields
        initial_state = PipelineState(
            # Original inputs (immutable)
            original_task=task,
            available_robots=available_robots,
            objects=objects,  # Now properly passed from main
            ground_truth=ground_truth,
            trans=trans,
            max_trans=max_trans,
            floor_plan=floor_plan,
            
            # Training examples (immutable)
            decomp_examples=examples[0],  # decomp_examples
            alloc_examples=examples[1],   # alloc_examples
            codegen_examples=examples[2], # codegen_examples
            
            # Work products (mutable)
            decomposed_plan="",
            allocation_plan="",
            generated_code="",
            
            # Process control
            current_stage="decomposition",
            iteration_count=0,
            max_iterations=25,
            
            # Agent iteration tracking
            decomposition_attempts=0,
            allocation_attempts=0,
            code_generation_attempts=0,
            max_attempts_per_agent=5,
            
            # Feedback system
            feedback_messages=[],
            errors=[],
            validation_results={},
            
            # Success tracking
            is_successful=False,
            final_stage_reached=""
        )
        
        try:
            print("🚀 Starting LangGraph workflow execution...")
            final_state = self.workflow.invoke(initial_state)
            
            # Log final results
            print(f"\n📊 LangGraph Results:")
            print(f"   Final Stage: {final_state.get('final_stage_reached', 'unknown')}")
            print(f"   Total Iterations: {final_state.get('iteration_count', 0)}")
            print(f"   Decomposition Attempts: {final_state.get('decomposition_attempts', 0)}")
            print(f"   Allocation Attempts: {final_state.get('allocation_attempts', 0)}")
            print(f"   Code Generation Attempts: {final_state.get('code_generation_attempts', 0)}")
            print(f"   Success: {final_state.get('is_successful', False)}")
            print(f"   Errors: {len(final_state.get('errors', []))}")
            
            return final_state
            
        except Exception as e:
            print(f"❌ LangGraph workflow error: {e}")
            initial_state['current_stage'] = 'failed'
            initial_state['final_stage_reached'] = 'failed'
            initial_state['errors'].append(str(e))
            return initial_state


def create_langgraph_pipeline_claude(api_key, decomp_model="claude-sonnet-4-20250514", 
                                    alloc_model="claude-sonnet-4-20250514", 
                                    codegen_model="claude-sonnet-4-20250514"):
    """Create LangGraph SMART-LLM pipeline with Claude API."""
    return LangGraphSMARTLLMClaude(api_key, decomp_model, alloc_model, codegen_model)
