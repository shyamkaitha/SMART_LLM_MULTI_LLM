# SMART-LLM Multi-LLM Pipeline Framework

## 🚀 Enhanced Multi-LLM Architecture

This repository contains an enhanced version of the SMART-LLM framework with a **3-stage Multi-LLM pipeline** that uses separate Large Language Models for each planning stage, providing better specialization and improved performance.

## ✨ Key Features

### **Multi-LLM Architecture**
- **Stage 1 - Task Decomposition**: Dedicated LLM for breaking down complex tasks
- **Stage 2 - Task Allocation**: Specialized LLM for robot assignment and coordination  
- **Stage 3 - Code Generation**: Focused LLM for generating executable AI2Thor code

### **Enhanced Capabilities**
- ✅ **Universal Evaluation System**: Fixed BROKEN state detection and case-insensitive object matching
- ✅ **Improved Training Examples**: Comprehensive examples for all action types including BreakObject
- ✅ **Better Context Management**: Optimized token usage and context length handling
- ✅ **Parallel Processing**: Enhanced multi-robot coordination and task execution
- ✅ **Backward Compatibility**: Works alongside the original single-LLM framework

## 🏗️ Architecture Comparison

| Feature | Single-LLM | Multi-LLM |
|---------|------------|-----------|
| **Planning Stages** | 1 LLM for all stages | 3 specialized LLMs |
| **Context Management** | Shared context | Optimized per stage |
| **Model Flexibility** | Single model | Different models per stage |
| **Specialization** | General purpose | Task-specific optimization |
| **Performance** | Good | Enhanced |

## 🚀 Quick Start

### **1. Setup Environment**
```bash
# Clone the repository
git clone https://github.com/shyamkaitha/SMART-LLM-MultiLLM.git
cd SMART-LLM-MultiLLM

# Install dependencies
pip install -r requirements.txt

# Set up API key
echo "your-openai-api-key" > api_key.txt
```

### **2. Run Multi-LLM Pipeline**
```bash
# Basic usage with GPT-4 for all stages
python3 scripts/multi_llm_pipeline.py --floor-plan FloorPlan209 --test-set "Break a vase and turn on the TV"

# Custom models per stage
python3 scripts/multi_llm_pipeline.py \
    --floor-plan FloorPlan209 \
    --decomp-model gpt-4 \
    --alloc-model gpt-4-turbo-preview \
    --codegen-model gpt-3.5-turbo \
    --test-set "Break a vase and turn on the TV"

# Execute the generated plan
python3 scripts/execute_plan.py --command "Break_a_vase_and_Turn_on_TV_plans_YYYY-MM-DD-HH-MM-SS"
```

### **3. Run Original Single-LLM (Still Available)**
```bash
python3 scripts/run_llm.py --floor-plan FloorPlan209 --test-set "Break a vase and turn on the TV"
```

## 📁 Key Files

### **Multi-LLM Pipeline**
- `scripts/multi_llm_pipeline.py` - Main multi-LLM pipeline implementation
- `data/pythonic_plans/train_task_decompose.py` - Task decomposition examples
- `data/pythonic_plans/train_task_allocation_solution.py` - Task allocation examples  
- `data/pythonic_plans/train_task_allocation_code.py` - Code generation examples

### **Enhanced Evaluation**
- `data/aithor_connect/end_thread.py` - Universal evaluation system with BROKEN state support

### **Original Framework**
- `scripts/run_llm.py` - Original single-LLM pipeline
- `scripts/execute_plan.py` - Plan execution (compatible with both frameworks)
- `scripts/ai2_thor_controller.py` - AI2Thor environment controller

## 🎯 Supported Tasks

The framework supports all original SMART-LLM tasks with enhanced performance:

- **Object Manipulation**: Pickup, Put, Drop, Throw
- **Object Interaction**: Open, Close, Switch On/Off
- **Object Modification**: Break, Slice, Cook
- **Complex Multi-Robot Tasks**: Parallel execution and coordination
- **Navigation**: GoToObject with optimized pathfinding

## 🔧 Configuration

### **Model Selection**
```bash
# Available models
--decomp-model: gpt-4, gpt-3.5-turbo, gpt-4-turbo-preview
--alloc-model: gpt-4, gpt-3.5-turbo, gpt-4-turbo-preview  
--codegen-model: gpt-4, gpt-3.5-turbo, gpt-4-turbo-preview
```

### **Floor Plans**
- `FloorPlan6.json` - Kitchen environment
- `FloorPlan15.json` - Living room setup
- `FloorPlan201.json` - Office space
- `FloorPlan209.json` - Multi-room layout
- `FloorPlan303.json` - Complex environment
- `FloorPlan414.json` - Bathroom setup

## 📊 Performance Improvements

### **Evaluation Metrics**
- **GCR (Goal Completion Rate)**: Enhanced with BROKEN state detection
- **TC (Task Completion)**: Improved accuracy
- **SR (Success Rate)**: Better overall success
- **Exec (Execution)**: More reliable execution
- **RU (Resource Utilization)**: Optimized robot usage

### **Multi-LLM Benefits**
- **Better Specialization**: Each LLM optimized for its specific stage
- **Improved Context Management**: Reduced token waste and context length issues
- **Enhanced Parallel Processing**: Better multi-robot coordination
- **Flexible Model Selection**: Use different models based on stage requirements

## 🐛 Bug Fixes

### **Universal Evaluation Fixes**
- ✅ Added missing `BROKEN` state detection
- ✅ Implemented case-insensitive object name matching
- ✅ Fixed container/receptacle object recognition
- ✅ Enhanced compatibility across all floor plans

### **Training Data Improvements**
- ✅ Added comprehensive `BreakObject` examples
- ✅ Enhanced multi-robot coordination examples
- ✅ Fixed parameter passing format for threading
- ✅ Improved action sequence generation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is based on the original SMART-LLM framework and maintains the same licensing terms.

## 🙏 Acknowledgments

- Original SMART-LLM framework by SMARTlab-Purdue
- AI2Thor simulation environment
- OpenAI GPT models for language processing
- Multi-LLM enhancements by the development team

## 📞 Support

For issues and questions:
- Create an issue in this repository
- Check the original SMART-LLM documentation
- Review the training examples for guidance

---

**Multi-LLM Framework**: Enhanced 3-stage planning with specialized LLMs for improved robot task execution and coordination.
