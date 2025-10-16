# main.py
import os
from dotenv import load_dotenv
from src.rag.rag_system import RAGSystem

def main():
    # 加载环境变量
    load_dotenv()
    
    # 初始化 RAG 系统
    rag_system = RAGSystem(data_path="./data", collection_name="insert_test_collection")
    
    # 检查系统状态
    system_info = rag_system.get_system_info()
    print("系统信息:", system_info)
    
    # 如果集合为空，构建索引
    # if not system_info.get("insert_test_collection", False):
    #     print("检测到空集合，开始构建文档索引...")
        # success = rag_system.build_index()
        # if not success:
        #     print("索引构建失败，请检查错误信息")
        #     return
    
    # 交互式问答循环
    print("\n=== RAG 问答系统 ===")
    print("输入 'quit' 或 '退出' 来结束程序")
    
    while True:
        try:
            question = input("\n请输入您的问题: ").strip()
            
            if question.lower() in ['quit', '退出', 'exit']:
                print("感谢使用！")
                break
            
            if not question:
                continue
            
            # 执行查询
            result = rag_system.query(question)
            
            # 显示结果
            print(f"\n🔍 问题: {result['question']}")
            print(f"🤖 回答: {result['answer']}")
            print(f"📊 置信度: {result['confidence']:.2f}")
            print(f"📚 参考文档: {len(result.get('retrieved_docs', []))} 个")
            
            if result.get('sources'):
                print("来源:")
                for source in result['sources'][:3]:  # 显示前3个来源
                    print(f"  - {source}")
                    
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"处理过程中出现错误: {e}")

if __name__ == "__main__":
    main()