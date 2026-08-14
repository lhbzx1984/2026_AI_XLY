"""
舌诊配套问卷模块
================
在舌象分析完成后收集用户症状、生活习惯、既往病史，
与舌象结果一起送入大模型进行综合分析。

问卷分三部分：
1. 基础症状（8题）：口干口苦、睡眠、食欲、大便、疲劳、手足温度、情绪、怕冷怕热
2. 生活习惯（6题）：出汗、饮食偏好、饮水、运动、作息、压力
3. 既往病史（5题）：高血压、糖尿病、胃肠疾病、肝胆疾病、家族病史

每个选项携带中医证素权重，用于辅助体质判断。
"""

# 证素权重键说明：
# qi_def=气虚, yang_def=阳虚, yin_def=阴虚, blood_def=血虚
# damp=痰湿, damp_heat=湿热, qi_stag=气滞, blood_stag=血瘀
# heat=化热, normal=正常

QUESTIONNAIRE = [
    # ===== 基础症状 =====
    {
        "section": "基础症状",
        "id": "mouth",
        "question": "口干口苦情况？",
        "options": ["无异常", "口干不苦", "口苦不干", "口干且苦"],
        "weights": [
            {"normal": 1},
            {"yin_def": 1},
            {"damp_heat": 1, "heat": 1},
            {"yin_def": 1, "damp_heat": 1, "heat": 1},
        ],
    },
    {
        "section": "基础症状",
        "id": "sleep",
        "question": "睡眠质量如何？",
        "options": ["良好（7-8小时）", "入睡困难", "易醒多梦", "失眠严重/早醒"],
        "weights": [
            {"normal": 1},
            {"yin_def": 1, "qi_stag": 1},
            {"yin_def": 1, "blood_def": 1},
            {"yin_def": 2, "heat": 1},
        ],
    },
    {
        "section": "基础症状",
        "id": "appetite",
        "question": "食欲情况？",
        "options": ["正常", "食欲不振", "食欲亢进/易饿", "饥不欲食"],
        "weights": [
            {"normal": 1},
            {"qi_def": 1, "damp": 1},
            {"heat": 1},
            {"yin_def": 1, "qi_def": 1},
        ],
    },
    {
        "section": "基础症状",
        "id": "bowel",
        "question": "大便情况？",
        "options": ["正常成形", "便溏/腹泻", "便秘干结", "时干时稀"],
        "weights": [
            {"normal": 1},
            {"damp": 1, "yang_def": 1},
            {"heat": 1, "yin_def": 1},
            {"qi_stag": 1, "damp": 1},
        ],
    },
    {
        "section": "基础症状",
        "id": "fatigue",
        "question": "疲劳程度？",
        "options": ["精力充沛", "稍感疲劳", "容易疲劳", "极度疲乏"],
        "weights": [
            {"normal": 1},
            {"qi_def": 1},
            {"qi_def": 2, "yang_def": 1},
            {"qi_def": 3, "blood_def": 1},
        ],
    },
    {
        "section": "基础症状",
        "id": "hand_foot",
        "question": "手脚温度？",
        "options": ["温暖适中", "手脚偏凉", "手脚心发热", "下肢冷上身热"],
        "weights": [
            {"normal": 1},
            {"yang_def": 2},
            {"yin_def": 2, "heat": 1},
            {"yin_def": 1, "yang_def": 1, "heat": 1},
        ],
    },
    {
        "section": "基础症状",
        "id": "mood",
        "question": "情绪状态？",
        "options": ["平和稳定", "容易烦躁", "情绪低落", "焦虑紧张"],
        "weights": [
            {"normal": 1},
            {"qi_stag": 2, "heat": 1},
            {"qi_stag": 2, "qi_def": 1},
            {"qi_stag": 2, "yin_def": 1},
        ],
    },
    {
        "section": "基础症状",
        "id": "thirst",
        "question": "怕冷还是怕热？",
        "options": ["冷热适中", "怕冷", "怕热", "又怕冷又怕热"],
        "weights": [
            {"normal": 1},
            {"yang_def": 2},
            {"heat": 2, "yin_def": 1},
            {"yin_def": 1, "yang_def": 1},
        ],
    },

    # ===== 生活习惯 =====
    {
        "section": "生活习惯",
        "id": "sweat",
        "question": "出汗情况？",
        "options": ["正常出汗", "动则多汗", "盗汗（夜间）", "很少出汗"],
        "weights": [
            {"normal": 1},
            {"qi_def": 2},
            {"yin_def": 2, "heat": 1},
            {"blood_stag": 1, "yin_def": 1},
        ],
    },
    {
        "section": "生活习惯",
        "id": "diet_pref",
        "question": "饮食偏好？",
        "options": ["饮食均衡", "嗜辛辣油腻", "嗜生冷寒凉", "嗜甜食"],
        "weights": [
            {"normal": 1},
            {"damp_heat": 1, "heat": 1},
            {"damp": 1, "yang_def": 1},
            {"damp": 2},
        ],
    },
    {
        "section": "生活习惯",
        "id": "water",
        "question": "饮水习惯？",
        "options": ["适量温水", "很少喝水", "爱喝冷水", "爱喝热饮"],
        "weights": [
            {"normal": 1},
            {"damp": 1, "yin_def": 1},
            {"heat": 1},
            {"yang_def": 1},
        ],
    },
    {
        "section": "生活习惯",
        "id": "exercise",
        "question": "运动频率？",
        "options": ["每周3次以上", "偶尔运动", "很少运动", "久坐不动"],
        "weights": [
            {"normal": 1},
            {"normal": 1},
            {"qi_def": 1, "blood_stag": 1},
            {"qi_def": 2, "blood_stag": 2, "damp": 1},
        ],
    },
    {
        "section": "生活习惯",
        "id": "sleep_time",
        "question": "通常几点入睡？",
        "options": ["23点前", "23-1点", "1-3点", "3点后或熬夜"],
        "weights": [
            {"normal": 1},
            {"yin_def": 1},
            {"yin_def": 2, "heat": 1},
            {"yin_def": 3, "heat": 1, "qi_def": 1},
        ],
    },
    {
        "section": "生活习惯",
        "id": "stress",
        "question": "压力程度？",
        "options": ["轻松", "偶尔有压力", "经常有压力", "压力很大"],
        "weights": [
            {"normal": 1},
            {"qi_stag": 1},
            {"qi_stag": 2, "qi_def": 1},
            {"qi_stag": 3, "heat": 1, "yin_def": 1},
        ],
    },

    # ===== 既往病史 =====
    {
        "section": "既往病史",
        "id": "hypertension",
        "question": "是否有高血压？",
        "options": ["无", "有（已控制）", "有（未控制）", "不确定"],
        "weights": [{}, {"blood_stag": 1, "heat": 1}, {"blood_stag": 2, "heat": 1, "yin_def": 1}, {}],
    },
    {
        "section": "既往病史",
        "id": "diabetes",
        "question": "是否有糖尿病？",
        "options": ["无", "有（已控制）", "有（未控制）", "不确定"],
        "weights": [{}, {"yin_def": 2, "heat": 1}, {"yin_def": 3, "heat": 2, "blood_stag": 1}, {}],
    },
    {
        "section": "既往病史",
        "id": "gi_disease",
        "question": "是否有胃肠疾病（胃炎/溃疡/肠炎）？",
        "options": ["无", "有（已控制）", "有（未控制）", "不确定"],
        "weights": [{}, {"qi_def": 1, "damp": 1}, {"qi_def": 2, "damp": 2, "damp_heat": 1}, {}],
    },
    {
        "section": "既往病史",
        "id": "liver_disease",
        "question": "是否有肝胆疾病（脂肪肝/胆囊炎/肝炎）？",
        "options": ["无", "有（已控制）", "有（未控制）", "不确定"],
        "weights": [{}, {"qi_stag": 1, "damp_heat": 1}, {"qi_stag": 2, "damp_heat": 2, "blood_stag": 1}, {}],
    },
    {
        "section": "既往病史",
        "id": "family_history",
        "question": "家族中有重大疾病史吗？",
        "options": ["无", "心血管疾病", "肿瘤", "代谢性疾病"],
        "weights": [{}, {"blood_stag": 1}, {"blood_stag": 1, "qi_stag": 1}, {"damp": 1, "yin_def": 1}],
    },
]

# 证素中文映射
ZHENG_SU_NAMES = {
    "normal": "正常", "qi_def": "气虚", "yang_def": "阳虚", "yin_def": "阴虚",
    "blood_def": "血虚", "damp": "痰湿", "damp_heat": "湿热",
    "qi_stag": "气滞", "blood_stag": "血瘀", "heat": "化热",
}


def compute_zheng_su(answers: dict) -> dict:
    """
    根据问卷答案计算证素权重。

    参数:
        answers: {question_id: option_index}

    返回:
        {证素key: 权重分, ...}（按分值降序）
    """
    scores = {}
    for q in QUESTIONNAIRE:
        qid = q["id"]
        if qid not in answers:
            continue
        opt_idx = answers[qid]
        if opt_idx < 0 or opt_idx >= len(q["weights"]):
            continue
        weights = q["weights"][opt_idx]
        for k, v in weights.items():
            scores[k] = scores.get(k, 0) + v
    # 按分值降序
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


def format_questionnaire_for_llm(answers: dict) -> str:
    """
    将问卷答案格式化为文本，供大模型输入。

    参数:
        answers: {question_id: option_index}

    返回:
        格式化的问卷文本
    """
    if not answers:
        return "（用户未填写问卷）"

    lines = []
    current_section = ""
    for q in QUESTIONNAIRE:
        qid = q["id"]
        if qid not in answers:
            continue
        if q["section"] != current_section:
            current_section = q["section"]
            lines.append(f"\n【{current_section}】")
        opt_idx = answers[qid]
        answer = q["options"][opt_idx] if 0 <= opt_idx < len(q["options"]) else "未回答"
        lines.append(f"  {q['question']} → {answer}")

    zheng_su = compute_zheng_su(answers)
    top_zheng_su = [f"{ZHENG_SU_NAMES.get(k, k)}({v}分)" for k, v in list(zheng_su.items())[:5] if k != "normal"]
    if top_zheng_su:
        lines.append(f"\n【问卷证素推断】{'、'.join(top_zheng_su)}")

    return "\n".join(lines)


def get_gradio_components():
    """
    生成 Gradio 问卷组件定义（供 main.py 使用）。

    返回:
        (组件列表, 组件ID映射)
    """
    components = []
    id_map = {}
    for q in QUESTIONNAIRE:
        comp = {
            "id": q["id"],
            "section": q["section"],
            "question": q["question"],
            "choices": q["options"],
        }
        components.append(comp)
        id_map[q["id"]] = len(components) - 1
    return components, id_map
