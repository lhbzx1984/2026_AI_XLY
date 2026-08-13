"""
中医舌象知识库模块
==================
本模块包含中医舌诊的核心理论知识，包括舌质分类、舌苔分类、
体质映射关系与健康科普建议。

重要声明：本知识库仅用于教育科普目的，不构成医疗诊断或治疗建议。
所有健康提示仅供参考，如有健康问题请咨询专业中医师。
"""

# ============================================================
# 舌质（舌体）分类 - 反映脏腑气血盛衰
# ============================================================
TONGUE_BODY_TYPES = {
    "pale_red": {
        "name": "淡红舌",
        "name_en": "Pale Red Tongue",
        "rgb_range": {"r": (180, 230), "g": (140, 190), "b": (130, 175)},
        "description": "舌色淡红而润，是正常舌象",
        "tcm_meaning": "气血充盈，脏腑功能正常",
        "health_status": "正常",
        "advice": "保持良好的生活习惯与饮食规律，继续坚持适度运动。",
    },
    "pale": {
        "name": "淡白舌",
        "name_en": "Pale Tongue",
        "rgb_range": {"r": (200, 250), "g": (195, 240), "b": (190, 235)},
        "description": "舌色较正常浅淡，甚至全无血色",
        "tcm_meaning": "气血两虚或阳虚",
        "health_status": "需关注",
        "advice": "可能提示气血不足，建议注意营养均衡，适当食用补气血的食物（如红枣、桂圆），保证充足睡眠。如持续不改善，建议咨询中医师。",
    },
    "red": {
        "name": "红舌",
        "name_en": "Red Tongue",
        "rgb_range": {"r": (180, 230), "g": (100, 150), "b": (90, 140)},
        "description": "舌色较正常红，甚至呈鲜红色",
        "tcm_meaning": "热证（实热或阴虚内热）",
        "health_status": "需关注",
        "advice": "可能提示体内有热，建议饮食清淡，少食辛辣油腻，多饮水，避免熬夜。学业压力大时注意适当休息放松。",
    },
    "crimson": {
        "name": "绛舌",
        "name_en": "Crimson Tongue",
        "rgb_range": {"r": (150, 200), "g": (60, 110), "b": (60, 110)},
        "description": "舌色深红，较红舌更深",
        "tcm_meaning": "热盛伤阴或阴虚火旺",
        "health_status": "建议就医",
        "advice": "可能提示热盛伤阴，建议避免辛辣刺激食物，注意养阴。如伴有其他不适，建议及时就医。",
    },
    "purple": {
        "name": "青紫舌",
        "name_en": "Purple Tongue",
        "rgb_range": {"r": (120, 170), "g": (90, 140), "b": (120, 170)},
        "description": "舌色发青或带有紫蓝色斑点",
        "tcm_meaning": "血瘀或气滞血瘀",
        "health_status": "建议就医",
        "advice": "可能提示气血运行不畅，建议适当运动促进血液循环，注意保暖。如持续存在，建议咨询中医师。",
    },
}

# ============================================================
# 舌苔分类 - 反映胃气盛衰和病邪性质
# ============================================================
TONGUE_COATING_TYPES = {
    "thin_white": {
        "name": "薄白苔",
        "name_en": "Thin White Coating",
        "description": "舌苔薄薄一层，色白，透过苔可看到舌体",
        "tcm_meaning": "正常苔象或表证初起",
        "health_status": "正常",
        "advice": "舌苔正常，说明胃气充盛。保持现有饮食习惯即可。",
    },
    "thick_white": {
        "name": "厚白苔",
        "name_en": "Thick White Coating",
        "description": "舌苔厚白，不能透过苔看到舌体",
        "tcm_meaning": "寒湿或痰湿内阻",
        "health_status": "需关注",
        "advice": "可能提示体内有寒湿或痰湿，建议少食生冷寒凉食物，适当食用健脾化湿的食物（如薏米、山药）。",
    },
    "yellow": {
        "name": "黄苔",
        "name_en": "Yellow Coating",
        "description": "舌苔呈黄色",
        "tcm_meaning": "热证或里证",
        "health_status": "需关注",
        "advice": "可能提示体内有热，建议饮食清淡，多食蔬菜水果，少食煎炸辛辣，注意充分饮水。",
    },
    "greasy_yellow": {
        "name": "黄腻苔",
        "name_en": "Greasy Yellow Coating",
        "description": "舌苔黄而粘腻",
        "tcm_meaning": "湿热蕴结",
        "health_status": "建议就医",
        "advice": "可能提示湿热蕴结，建议饮食以清淡为主，避免甜腻食物。如伴有口苦、食欲不振等，建议就医。",
    },
    "peeled": {
        "name": "剥苔/无苔",
        "name_en": "Peeled/No Coating",
        "description": "舌苔部分剥脱或完全无苔",
        "tcm_meaning": "胃阴不足或气阴两虚",
        "health_status": "需关注",
        "advice": "可能提示胃阴不足，建议注意养阴润燥，多食滋阴食物（如银耳、百合），避免过度劳累。",
    },
    "gray_black": {
        "name": "灰黑苔",
        "name_en": "Gray-Black Coating",
        "description": "舌苔呈灰色或黑色",
        "tcm_meaning": "寒极或热极",
        "health_status": "建议就医",
        "advice": "灰黑苔较为特殊，建议尽早就医检查，由专业医师判断。",
    },
}

# ============================================================
# 体质类型映射（基于舌象综合判断）
# ============================================================
CONSTITUTION_MAP = {
    # (舌质, 舌苔) -> 体质类型
    ("pale_red", "thin_white"): {
        "type": "平和质",
        "description": "体形匀称健壮，面色润泽，精力充沛",
        "feature": "正常体质，阴阳气血调和",
    },
    ("pale", "thin_white"): {
        "type": "气虚质",
        "description": "容易疲乏，气短懒言，易感冒",
        "feature": "元气不足，脏腑功能偏低",
    },
    ("pale", "thick_white"): {
        "type": "阳虚质",
        "description": "怕冷，手足不温，喜热饮食",
        "feature": "阳气不足，虚寒内生",
    },
    ("red", "yellow"): {
        "type": "湿热质",
        "description": "面垢油光，口苦，易生痤疮",
        "feature": "湿热内蕴，气机不畅",
    },
    ("red", "thin_white"): {
        "type": "阴虚质",
        "description": "手足心热，口燥咽干，喜冷饮",
        "feature": "阴液亏少，虚热内生",
    },
    ("purple", "thin_white"): {
        "type": "血瘀质",
        "description": "肤色晦暗，易有瘀斑，唇色偏暗",
        "feature": "血行不畅，瘀血内阻",
    },
    ("pale_red", "thick_white"): {
        "type": "痰湿质",
        "description": "体形肥胖，腹部松软，口黏腻",
        "feature": "水液内停，痰湿凝聚",
    },
}

# 默认体质（无法精确匹配时）
DEFAULT_CONSTITUTION = {
    "type": "混合体质",
    "description": "舌象特征介于多种体质之间",
    "feature": "建议综合参考各项指标，咨询专业中医师进行体质辨识",
}

# ============================================================
# 拍摄指南
# ============================================================
PHOTO_GUIDE = [
    "请在自然光下拍摄（避免黄色灯光或强烈直射光）",
    "充分伸舌，舌尖自然下垂",
    "嘴巴张大，确保舌体完整可见",
    "距离摄像头约 15-20 厘米",
    "拍摄前 30 分钟避免进食有色食物（如咖啡、糖果）",
    "保持相机稳定，避免模糊",
]


def get_tongue_body_info(body_key: str) -> dict:
    """根据舌质类型键获取详细信息"""
    return TONGUE_BODY_TYPES.get(body_key, TONGUE_BODY_TYPES["pale_red"])


def get_coating_info(coating_key: str) -> dict:
    """根据舌苔类型键获取详细信息"""
    return TONGUE_COATING_TYPES.get(coating_key, TONGUE_COATING_TYPES["thin_white"])


def get_constitution(body_key: str, coating_key: str) -> dict:
    """
    根据舌质和舌苔综合判断体质类型
    返回体质名称、描述与特征
    """
    return CONSTITUTION_MAP.get((body_key, coating_key), DEFAULT_CONSTITUTION)


def get_health_advice(body_key: str, coating_key: str) -> str:
    """生成综合健康科普建议（非诊断）"""
    body_info = get_tongue_body_info(body_key)
    coating_info = get_coating_info(coating_key)

    advice_parts = [
        f"【舌质分析】{body_info['name']}：{body_info['tcm_meaning']}",
        f"【舌苔分析】{coating_info['name']}：{coating_info['tcm_meaning']}",
        "",
        f"【健康提示】{body_info['advice']}",
        f"{coating_info['advice']}",
    ]
    return "\n".join(advice_parts)


# ============================================================
# 免责声明（必须在应用中展示）
# ============================================================
DISCLAIMER = """⚠️ 重要免责声明

本项目为教育演示用途，非医疗器械，不用于临床诊断。
本项目不提供疾病诊断、治疗或用药建议，所有健康提示仅供参考。
如有健康问题请咨询专业中医师。
本项目不采集您的身份信息，舌象图像仅在本地处理，不会上传云端。
未成年人使用前须获得监护人同意。"""
