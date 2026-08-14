"""
中医舌象图谱分类与知识映射模块
================================
对接 YOLO 21 类舌象标注数据集与项目知识库。

21 类分类体系：
  - 舌质整体（0-3）：健康舌、薄苔舌、红舌、紫舌
  - 舌质形态（4-8）：胖大舌、瘦舌、红点舌、裂纹舌、齿痕舌
  - 舌苔（9-12）：白苔、黄苔、黑苔、花苔
  - 舌面脏腑分区凹凸（13-20）：肾/肝胆/脾胃/心肺 区的凹陷与凸起

舌面脏腑对应（中医舌诊理论）：
  - 舌尖 → 心肺
  - 舌中 → 脾胃
  - 舌根 → 肾
  - 舌边 → 肝胆
  凹 = 相应脏腑正气虚；凸 = 相应脏腑邪气实

重要声明：本模块仅用于教育科普目的，不构成医疗诊断。
"""

# ============================================================
# YOLO 21 类完整定义
# ============================================================
YOLO_CLASSES = {
    0:  {"pinyin": "jiankangshe",  "name": "健康舌",   "category": "整体", "kb_keys": ["pale_red", "thin_white"],       "tcm_meaning": "气血充盈，脏腑功能正常",            "advice": "保持良好生活习惯，规律作息，适度运动。"},
    1:  {"pinyin": "botaishe",     "name": "薄苔舌",   "category": "舌苔", "kb_keys": ["thin_white"],                    "tcm_meaning": "正常苔象或表证初起",                "advice": "舌苔正常，保持饮食规律。若初感外邪注意休息。"},
    2:  {"pinyin": "hongshe",      "name": "红舌",     "category": "舌质", "kb_keys": ["red"],                           "tcm_meaning": "热证（实热或阴虚内热）",            "advice": "饮食清淡，少食辛辣，多饮水，避免熬夜。"},
    3:  {"pinyin": "zishe",        "name": "紫舌",     "category": "舌质", "kb_keys": ["purple"],                        "tcm_meaning": "血瘀或气滞血瘀",                    "advice": "适当运动促进血液循环，注意保暖，避免久坐。"},
    4:  {"pinyin": "pangdashe",    "name": "胖大舌",   "category": "形态", "kb_keys": [],                                "tcm_meaning": "脾虚湿盛或阳虚水泛",                "advice": "健脾利湿，少食生冷，适当食用薏米、山药、茯苓。"},
    5:  {"pinyin": "shoushe",      "name": "瘦舌",     "category": "形态", "kb_keys": [],                                "tcm_meaning": "气血不足或阴虚火旺",                "advice": "补益气血或滋阴，注意营养均衡，避免过度劳累。"},
    6:  {"pinyin": "hongdianshe",  "name": "红点舌",   "category": "特征", "kb_keys": [],                                "tcm_meaning": "热毒炽盛或血热",                    "advice": "清热解毒，多食清凉食物，少食燥热辛辣之物。"},
    7:  {"pinyin": "liewenshe",    "name": "裂纹舌",   "category": "特征", "kb_keys": [],                                "tcm_meaning": "阴虚液损或血虚",                    "advice": "滋阴润燥，多食银耳、百合、梨，避免辛辣。"},
    8:  {"pinyin": "chihenshe",    "name": "齿痕舌",   "category": "特征", "kb_keys": [],                                "tcm_meaning": "脾虚或气虚",                        "advice": "健脾益气，少食生冷，适当食用山药、大枣、扁豆。"},
    9:  {"pinyin": "baitaishe",    "name": "白苔",     "category": "舌苔", "kb_keys": ["thin_white", "thick_white"],     "tcm_meaning": "正常苔象或寒证、湿证",              "advice": "薄白为正常；厚白提示寒湿，宜温中健脾，少食生冷。"},
    10: {"pinyin": "huangtaishe",  "name": "黄苔",     "category": "舌苔", "kb_keys": ["yellow", "greasy_yellow"],       "tcm_meaning": "热证或里证",                        "advice": "饮食清淡，多食蔬果，少食煎炸辛辣，注意饮水。"},
    11: {"pinyin": "heitaishe",    "name": "黑苔",     "category": "舌苔", "kb_keys": ["gray_black"],                    "tcm_meaning": "寒极或热极",                        "advice": "灰黑苔较为特殊，建议尽早就医检查。"},
    12: {"pinyin": "huataishe",    "name": "花苔",     "category": "舌苔", "kb_keys": ["peeled"],                        "tcm_meaning": "胃阴不足或气阴两虚",                "advice": "养阴润燥，多食银耳、百合，避免过度劳累。"},
    13: {"pinyin": "shenquao",     "name": "肾区凹",   "category": "分区", "kb_keys": [],                                "tcm_meaning": "肾精不足或肾阴虚",                  "advice": "补肾益精，注意休息，避免房劳过度，适当食用黑芝麻、核桃。"},
    14: {"pinyin": "shenqutu",     "name": "肾区凸",   "category": "分区", "kb_keys": [],                                "tcm_meaning": "肾邪气实或湿热下注",                "advice": "清热利湿，少食肥甘厚味，建议咨询中医师。"},
    15: {"pinyin": "gandanao",     "name": "肝胆凹",   "category": "分区", "kb_keys": [],                                "tcm_meaning": "肝血不足或肝阴亏虚",                "advice": "养血柔肝，保证睡眠，适当食用枸杞、菠菜。"},
    16: {"pinyin": "gandantu",     "name": "肝胆凸",   "category": "分区", "kb_keys": [],                                "tcm_meaning": "肝郁气滞或肝胆湿热",                "advice": "疏肝理气，调节情绪，少食油腻，可饮菊花茶。"},
    17: {"pinyin": "piweiao",      "name": "脾胃凹",   "category": "分区", "kb_keys": [],                                "tcm_meaning": "脾胃虚弱或胃阴不足",                "advice": "健脾养胃，定时定量进食，适当食用山药、小米粥。"},
    18: {"pinyin": "piweitu",      "name": "脾胃凸",   "category": "分区", "kb_keys": [],                                "tcm_meaning": "脾胃湿热或食积",                    "advice": "清热化湿，消食导滞，少食甜腻，适当运动。"},
    19: {"pinyin": "xinfeiao",     "name": "心肺凹",   "category": "分区", "kb_keys": [],                                "tcm_meaning": "心肺气虚或阴虚",                    "advice": "补益心肺，避免剧烈运动，适当食用百合、莲子。"},
    20: {"pinyin": "xinfeitu",     "name": "心肺凸",   "category": "分区", "kb_keys": [],                                "tcm_meaning": "心肺邪热或痰热",                    "advice": "清心化痰，少食辛辣燥热，保持心情平和。"},
}

# ============================================================
# 舌面脏腑分区定义（位置先验）
# ============================================================
# 舌面四区：按归一化坐标划分（y 方向：上=舌尖，下=舌根）
#   舌尖（心肺）：y < 0.33
#   舌中（脾胃）：0.33 <= y < 0.66
#   舌根（肾）：  y >= 0.66
#   舌边（肝胆）：x < 0.33 或 x >= 0.67（在舌中范围内）
ORGAN_REGIONS = {
    "心肺": {"y_range": (0.0, 0.33), "x_range": (0.33, 0.67), "yolo_ids": [19, 20]},
    "脾胃": {"y_range": (0.33, 0.66), "x_range": (0.33, 0.67), "yolo_ids": [17, 18]},
    "肾":   {"y_range": (0.66, 1.0),  "x_range": (0.33, 0.67), "yolo_ids": [13, 14]},
    "肝胆": {"y_range": (0.33, 0.66), "x_range_left": (0.0, 0.33), "x_range_right": (0.67, 1.0), "yolo_ids": [15, 16]},
}

# 脏腑分区 → 疾病风险提示
ORGAN_DISEASE_RISK = {
    "心肺": {
        "deficiency": {"name": "心肺气阴两虚", "risks": ["心悸", "失眠", "气短", "上呼吸道易感"]},
        "excess":     {"name": "心肺痰热",     "risks": ["咳嗽痰多", "胸闷", "心烦", "口腔溃疡"]},
    },
    "脾胃": {
        "deficiency": {"name": "脾胃虚弱",     "risks": ["消化不良", "食欲不振", "腹胀", "慢性胃炎"]},
        "excess":     {"name": "脾胃湿热/食积","risks": ["口臭", "胃胀", "反酸", "消化性溃疡"]},
    },
    "肾": {
        "deficiency": {"name": "肾精不足",     "risks": ["腰膝酸软", "记忆力下降", "耳鸣", "性功能减退"]},
        "excess":     {"name": "湿热下注",     "risks": ["泌尿系感染", "下肢浮肿", "小便异常"]},
    },
    "肝胆": {
        "deficiency": {"name": "肝血/肝阴不足","risks": ["视物模糊", "眼睛干涩", "月经量少", "肢体麻木"]},
        "excess":     {"name": "肝郁/肝胆湿热","risks": ["情绪急躁", "胁肋胀痛", "脂肪肝", "胆囊炎"]},
    },
}


# ============================================================
# 新增舌质形态特征知识（知识库中没有的）
# ============================================================
TONGUE_SHAPE_FEATURES = {
    "pangda": {
        "name": "胖大舌",
        "description": "舌体较正常胖大，甚至充斥口腔",
        "tcm_meaning": "脾虚湿盛或阳虚水泛",
        "detection": "舌体宽度占图像比例偏大，边缘可能伴有齿痕",
    },
    "shou": {
        "name": "瘦舌",
        "description": "舌体较正常瘦小而薄",
        "tcm_meaning": "气血不足或阴虚火旺",
        "detection": "舌体宽度占图像比例偏小，舌体偏薄",
    },
    "hongdian": {
        "name": "红点舌",
        "description": "舌面出现红色点状突起（蕈状乳头充血）",
        "tcm_meaning": "热毒炽盛或血热",
        "detection": "舌面局部检测到红色高亮小区域",
    },
    "liewen": {
        "name": "裂纹舌",
        "description": "舌面出现深浅不一的裂纹",
        "tcm_meaning": "阴虚液损或血虚",
        "detection": "舌面检测到线状低亮度区域",
    },
    "chihen": {
        "name": "齿痕舌",
        "description": "舌体边缘有牙齿压迫的痕迹",
        "tcm_meaning": "脾虚或气虚",
        "detection": "舌体边缘呈波浪状不规则",
    },
}


# ============================================================
# 知识库 key → YOLO 类映射（反向映射，用于评估）
# ============================================================
KB_TO_YOLO_MAP = {
    # 舌质
    "pale_red":     [0],      # 淡红舌 → 健康舌
    "red":          [2],      # 红舌
    "crimson":      [2],      # 绛舌 → 也归入红舌大类
    "purple":       [3],      # 青紫舌 → 紫舌
    "pale":         [],       # 淡白舌 → 21类中无直接对应
    # 舌苔
    "thin_white":   [1, 9],   # 薄白苔 → 薄苔舌 + 白苔
    "thick_white":  [9],      # 厚白苔 → 白苔
    "yellow":       [10],     # 黄苔
    "greasy_yellow":[10],     # 黄腻苔 → 黄苔
    "peeled":       [12],     # 剥苔 → 花苔
    "gray_black":   [11],     # 灰黑苔 → 黑苔
}


def get_yolo_class_info(class_id: int) -> dict:
    """获取 YOLO 类别的完整信息"""
    return YOLO_CLASSES.get(class_id, {"pinyin": "unknown", "name": f"未知类{class_id}", "category": "未知", "kb_keys": [], "tcm_meaning": "", "advice": ""})


def map_kb_result_to_yolo(kb_body_key: str, kb_coating_key: str) -> list:
    """
    将知识库分析结果（舌质key + 舌苔key）映射为 YOLO 类别列表。

    参数:
        kb_body_key: 知识库舌质 key（如 "red"）
        kb_coating_key: 知识库舌苔 key（如 "yellow"）

    返回:
        YOLO 类 ID 列表
    """
    yolo_ids = []
    body_map = KB_TO_YOLO_MAP.get(kb_body_key, [])
    coating_map = KB_TO_YOLO_MAP.get(kb_coating_key, [])
    yolo_ids.extend(body_map)
    yolo_ids.extend(coating_map)
    # 去重
    return list(dict.fromkeys(yolo_ids))


def parse_yolo_label(label_path: str) -> list:
    """
    解析 YOLO 标注文件，返回标注列表。

    返回:
        [{"class": int, "x_center": float, "y_center": float, "width": float, "height": float}, ...]
    """
    annotations = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                annotations.append({
                    "class": int(parts[0]),
                    "x_center": float(parts[1]),
                    "y_center": float(parts[2]),
                    "width": float(parts[3]),
                    "height": float(parts[4]),
                })
    return annotations


def classify_annotation(annotation: dict) -> str:
    """
    判断标注框类型：整舌框 or 局部特征框。

    YOLO 标注中：
    - 整舌框：width 和 height 较大（> 0.5），标注整体舌质/舌苔类别
    - 局部框：width 或 height 较小（< 0.2），标注局部特征（红点、裂纹、分区）
    """
    w, h = annotation["width"], annotation["height"]
    if w > 0.5 and h > 0.5:
        return "whole"
    elif w < 0.15 or h < 0.15:
        return "local"
    else:
        return "region"


def get_organ_region(y_center: float, x_center: float) -> str:
    """
    根据标注框中心坐标判断所属脏腑分区。

    返回: "心肺" / "脾胃" / "肾" / "肝胆" / "未知"
    """
    y, x = y_center, x_center
    if y < 0.33 and 0.33 <= x < 0.67:
        return "心肺"
    elif 0.33 <= y < 0.66:
        if 0.33 <= x < 0.67:
            return "脾胃"
        else:
            return "肝胆"
    elif y >= 0.66 and 0.33 <= x < 0.67:
        return "肾"
    return "未知"


def build_organ_analysis(annotations: list) -> dict:
    """
    从标注中提取脏腑分区凹凸信息。

    返回:
        {"心肺": "凹"/"凸"/None, "脾胃": ..., "肾": ..., "肝胆": ...}
    """
    result = {"心肺": None, "脾胃": None, "肾": None, "肝胆": None}
    for ann in annotations:
        cls = ann["class"]
        organ_map = {13: ("肾", "凹"), 14: ("肾", "凸"), 15: ("肝胆", "凹"),
                     16: ("肝胆", "凸"), 17: ("脾胃", "凹"), 18: ("脾胃", "凸"),
                     19: ("心肺", "凹"), 20: ("心肺", "凸")}
        if cls in organ_map:
            organ, state = organ_map[cls]
            result[organ] = state
    return result
