"""
初始化测试数据
- 创建示例员工信息 (employees.json)
- 创建示例公司文档 (PDF/TXT)
- 将文档索引到 ChromaDB
"""
import os
import json

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")


def create_employees_data():
    """创建示例员工数据"""
    employees = [
        {"name": "张三", "department": "技术部", "position": "高级工程师", "email": "zhangsan@company.com", "phone": "138-0001-0001"},
        {"name": "李四", "department": "技术部", "position": "前端工程师", "email": "lisi@company.com", "phone": "138-0001-0002"},
        {"name": "王五", "department": "产品部", "position": "产品经理", "email": "wangwu@company.com", "phone": "138-0001-0003"},
        {"name": "赵六", "department": "设计部", "position": "UI设计师", "email": "zhaoliu@company.com", "phone": "138-0001-0004"},
        {"name": "钱七", "department": "人事部", "position": "HR经理", "email": "qianqi@company.com", "phone": "138-0001-0005"},
        {"name": "孙八", "department": "财务部", "position": "财务主管", "email": "sunba@company.com", "phone": "138-0001-0006"},
        {"name": "周九", "department": "技术部", "position": "后端工程师", "email": "zhoujiu@company.com", "phone": "138-0001-0007"},
        {"name": "吴十", "department": "市场部", "position": "市场总监", "email": "wushi@company.com", "phone": "138-0001-0008"},
        {"name": "郑十一", "department": "技术部", "position": "架构师", "email": "zhengsy@company.com", "phone": "138-0001-0009"},
        {"name": "陈十二", "department": "产品部", "position": "产品助理", "email": "chense@company.com", "phone": "138-0001-0010"},
    ]

    output_path = os.path.join(DATA_DIR, "employees.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(employees, f, ensure_ascii=False, indent=2)

    print(f"✅ 员工数据已创建: {output_path} ({len(employees)} 人)")
    return employees


def create_sample_documents():
    """创建示例公司文档"""
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)

    docs = {
        "公司休假制度.txt": """智远科技有限公司 休假制度
版本：V2.0  生效日期：2025年1月1日

第一章 年假制度
1. 入职满1年的员工，可享受5天带薪年假
2. 入职满3年的员工，可享受10天带薪年假
3. 入职满5年的员工，可享受15天带薪年假
4. 年假当年有效，不可跨年累积，特殊情况需HR审批

第二章 病假制度
1. 员工因病需请假，须提供医院证明
2. 带薪病假每年5天，超出部分按事假处理
3. 连续病假超过3天需部门主管审批

第三章 事假制度
1. 事假需提前1天申请，紧急情况除外
2. 事假期间不计薪资
3. 每月事假不超过3天，超出需部门总监审批

第四章 婚假/产假
1. 婚假：法定婚假3天，晚婚加7天
2. 产假：女员工产假158天，男员工陪产假15天
3. 陪产假须在新生儿出生后3个月内使用

第五章 加班调休
1. 工作日加班可申请调休，比例为1:1
2. 周末加班调休比例为1:1.5
3. 法定节假日加班调休比例为1:2
4. 调休须在加班后3个月内使用""",
        "员工手册.txt": """智远科技有限公司 员工手册
版本：V3.0  生效日期：2025年3月1日

第一章 公司简介
智远科技有限公司成立于2018年，专注于人工智能和大数据领域，现有员工200余人，总部位于北京中关村科技园。

第二章 考勤规定
1. 工作时间：周一至周五 9:00-18:00
2. 弹性工时：核心工作时间为10:00-16:00，其余时间可弹性安排
3. 迟到规定：每月迟到3次以内不扣薪，超过3次每次扣50元
4. 打卡方式：支持指纹打卡和企业微信打卡

第三章 报销制度
1. 差旅报销：出差需提前申请，住宿标准一二线城市500元/晚，三四线城市300元/晚
2. 交通报销：市内交通实报实销，打车需说明原因
3. 餐费报销：出差期间餐补100元/天
4. 报销周期：每月1-5日提交，15日前到账

第四章 福利待遇
1. 五险一金：按实际薪资缴纳
2. 补充医疗保险：公司统一购买
3. 年度体检：每年一次免费体检
4. 团建活动：每季度一次部门团建，预算200元/人
5. 节日福利：春节、中秋等节日发放礼品

第五章 晋升制度
1. 晋升周期：每年1月和7月各一次晋升窗口
2. 晋升条件：绩效评级B+及以上，且在当前职级满1年
3. 晋升流程：个人申请 → 直属主管推荐 → 部门评审 → HR审批""",
        "技术部规范.txt": """智远科技有限公司 技术部工作规范
版本：V1.5  生效日期：2025年2月1日

第一章 代码规范
1. 代码必须通过Lint检查才能提交
2. 提交信息格式：[类型] 描述，如 [feat] 新增用户登录功能
3. 类型包括：feat(新功能)、fix(修复)、docs(文档)、refactor(重构)、test(测试)
4. 代码审查：所有代码必须至少一位同事Review后才能合并

第二章 Git 工作流
1. 分支策略：main(生产) → develop(开发) → feature/xxx(功能)
2. 功能开发在feature分支进行，完成后提PR合并到develop
3. 发版时从develop合并到main，打tag
4. 禁止直接push到main分支

第三章 部署流程
1. 开发环境：dev.company.com
2. 测试环境：staging.company.com
3. 生产环境：app.company.com
4. 部署方式：通过CI/CD自动部署，禁止手动操作生产服务器

第四章 技术栈
1. 后端：Python (FastAPI) + PostgreSQL + Redis
2. 前端：React + TypeScript + Tailwind CSS
3. 基础设施：Docker + Kubernetes + 阿里云
4. 监控：Prometheus + Grafana + Sentry""",
        "报销流程.txt": """智远科技有限公司 报销流程指南
版本：V2.1  生效日期：2025年1月15日

第一章 差旅报销
1. 出差申请：提前3天在OA系统提交出差申请
2. 出差审批：直属主管 → 部门总监 → 财务部
3. 住宿标准：
   - 一线城市：单间500元/晚以内
   - 二线城市：单间400元/晚以内
   - 三四线城市：单间300元/晚以内
4. 交通标准：
   - 飞机：经济舱
   - 高铁：二等座（4小时以内）、一等座（4小时以上）
   - 市内交通：实报实销

第二章 日常报销
1. 办公用品：部门统一采购，个人不超过200元可自行购买
2. 招待费用：需提前申请，标准200元/人以内
3. 培训费用：需提前申请，年度预算3000元/人

第三章 报销材料
1. 发票原件（电子发票需打印）
2. 费用明细表
3. 审批单据

第四章 报销时限
1. 费用发生后30天内提交报销
2. 超过30天需额外审批
3. 跨年费用不予报销""",
    }

    for filename, content in docs.items():
        file_path = os.path.join(DOCUMENTS_DIR, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ 示例文档已创建: {filename}")

    return list(docs.keys())


def main():
    print("=" * 50)
    print("🚀 初始化企业文档智能助手测试数据...")
    print("=" * 50)

    # 1. 创建数据目录
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)

    # 2. 创建员工数据
    employees = create_employees_data()

    # 3. 创建示例文档
    doc_names = create_sample_documents()

    print("\n" + "=" * 50)
    print(f"✅ 初始化完成！")
    print(f"   - 员工数据: {len(employees)} 人")
    print(f"   - 示例文档: {len(doc_names)} 个")
    print(f"\n💡 下一步:")
    print(f"   1. 复制 .env.example 为 .env 并填写 API Key")
    print(f"   2. 运行 python app/main.py 启动服务")
    print(f"   3. 打开 Gradio 界面上传文档或直接提问")
    print("=" * 50)


if __name__ == "__main__":
    main()