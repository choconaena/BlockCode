# blocks/ModelDesign/start.py
# 서브모듈에 의존하지 않고, form 값으로 CNN 클래스를 완성해 내보내는 버전

def _act(name: str) -> str:
    if not name:
        return "nn.ReLU()"
    n = name.strip().lower()
    if n == "relu": return "nn.ReLU()"
    if n == "tanh": return "nn.Tanh()"
    if n == "sigmoid": return "nn.Sigmoid()"
    if n == "softmax": return "nn.Softmax(dim=1)"
    if n == "none": return ""   # 비활성
    return "nn.ReLU()"

def _same_padding(k: int) -> int:
    # odd kernel 가정
    try:
        k = int(k)
    except:
        k = 3
    return k // 2

def _size_after_conv(size: int, k: int, padding: int, stride: int = 1) -> int:
    # floor((n + 2p - k)/s + 1), dilation=1
    return (size + 2*padding - k) // stride + 1

def _size_after_pool(size: int, k: int, stride: int) -> int:
    return (size - k) // stride + 1

def generate_modeldesign_snippet(form, connection_info=None):
    """
    form: request.form
    - 입력: input_w, input_h, input_c
    - conv1_* (필수), pool1_*, conv2_* (선택), use_conv2
    - use_dropout, dropout_p
    - dense_units, dense_activation, num_classes
    - resize_n (전처리에서 온 사이즈, 없으면 input_w/h 사용)
    """
    # ---- 입력/기본값 ----
    try:
        in_c = int(form.get("input_c", 1))
        in_w = int(form.get("input_w", 28))
        in_h = int(form.get("input_h", 28))
    except:
        in_c, in_w, in_h = 1, 28, 28

    # 전처리 단계에서 resize_n을 썼다면 그걸 기준으로 함
    try:
        size = int(form.get("resize_n", min(in_w, in_h)))
    except:
        size = min(in_w, in_h)

    # Conv1
    conv1_filters = int(form.get("conv1_filters", 0) or 0)
    conv1_kernel  = int(form.get("conv1_kernel", 3))
    conv1_pad_key = form.get("conv1_padding", "valid")
    conv1_pad     = _same_padding(conv1_kernel) if conv1_pad_key == "same" else 0
    conv1_act     = form.get("conv1_activation", "relu")

    # Pool1
    pool1_type   = form.get("pool1_type", "")  # "max"/"avg"/""
    pool1_size   = int(form.get("pool1_size", 2))
    pool1_stride = int(form.get("pool1_stride", pool1_size))

    # Conv2 (선택)
    use_conv2     = bool(form.get("use_conv2"))
    conv2_filters = int(form.get("conv2_filters", 0) or 0)
    conv2_kernel  = int(form.get("conv2_kernel", 3))
    # 패딩 선택이 UI에 없으므로 conv2는 same으로 가정(출력 크기 유지)
    conv2_pad     = _same_padding(conv2_kernel)
    conv2_act     = form.get("conv2_activation", "relu")

    # Dropout
    use_dropout = bool(form.get("use_dropout"))
    try:
        dropout_p = float(form.get("dropout_p", 0.25))
    except:
        dropout_p = 0.25

    # Dense / Output
    dense_units   = int(form.get("dense_units", 128))
    dense_act     = form.get("dense_activation", "relu")
    num_classes   = int(form.get("num_classes", 10))

    # ---- 모델 코드 생성 ----
    lines = []
    lines.append("import torch")
    lines.append("import torch.nn as nn")
    lines.append("")
    lines.append("class CNN(nn.Module):")
    lines.append("    def __init__(self):")
    lines.append("        super().__init__()")
    lines.append("")
    lines.append("        # 합성곱 블록")
    lines.append("        self.conv_layers = nn.Sequential(")

    current_c    = in_c
    current_size = size

    # Conv1
    if conv1_filters > 0:
        lines.append(f"            nn.Conv2d(in_channels={current_c}, out_channels={conv1_filters}, kernel_size={conv1_kernel}, padding={conv1_pad}),")
        if _act(conv1_act):
            lines.append(f"            {_act(conv1_act)},")
        # 출력 크기 업데이트
        current_size = _size_after_conv(current_size, conv1_kernel, conv1_pad, 1)
        # Pool1
        if pool1_type:
            pool_cls = "MaxPool2d" if pool1_type.lower() == "max" else "AvgPool2d"
            lines.append(f"            nn.{pool_cls}(kernel_size={pool1_size}, stride={pool1_stride}),")
            current_size = _size_after_pool(current_size, pool1_size, pool1_stride)
        current_c = conv1_filters

    # Conv2 (옵션)
    if use_conv2 and conv2_filters > 0:
        lines.append(f"            nn.Conv2d(in_channels={current_c}, out_channels={conv2_filters}, kernel_size={conv2_kernel}, padding={conv2_pad}),")
        if _act(conv2_act):
            lines.append(f"            {_act(conv2_act)},")
        # same padding 가정 → 크기 유지
        current_c = conv2_filters

    # Dropout (옵션)
    if use_dropout:
        lines.append(f"            nn.Dropout(p={dropout_p}),")

    lines.append("        )")
    lines.append("")
    lines.append("        # 완전연결 블록")
    fc_input = f"{current_c}*{current_size}*{current_size}"
    lines.append("        self.fc_layers = nn.Sequential(")
    lines.append("            nn.Flatten(),")
    lines.append(f"            nn.Linear({fc_input}, {dense_units}),")
    if _act(dense_act):
        lines.append(f"            {_act(dense_act)},")
    lines.append(f"            nn.Linear({dense_units}, {num_classes}),")
    lines.append("        )")
    lines.append("")
    lines.append("    def forward(self, x):")
    lines.append("        x = self.conv_layers(x)")
    lines.append("        x = self.fc_layers(x)")
    lines.append("        return x")
    lines.append("")

    return "\n".join(lines)

def calculate_connections(form, preprocessing_info=None):
    """
    (호환용) app.py가 호출하므로 남겨둠.
    conv2 in_channels, FC 입력 크기 등을 계산해 돌려준다.
    """
    try:
        in_c = int(form.get("input_c", 1))
        resize_n = int((preprocessing_info or {}).get("resize_n", form.get("resize_n", 28)))
    except:
        in_c, resize_n = 1, 28

    current_channels = in_c
    current_size     = resize_n

    conv1_filters = int(form.get("conv1_filters", 0) or 0)
    conv1_kernel  = int(form.get("conv1_kernel", 3))
    conv1_pad     = _same_padding(conv1_kernel) if form.get("conv1_padding", "valid") == "same" else 0

    # conv1
    if conv1_filters > 0:
        current_size     = _size_after_conv(current_size, conv1_kernel, conv1_pad, 1)
        current_channels = conv1_filters

        # pool1
        pool1_type   = form.get("pool1_type", "")
        pool1_size   = int(form.get("pool1_size", 2))
        pool1_stride = int(form.get("pool1_stride", pool1_size))
        if pool1_type:
            current_size = _size_after_pool(current_size, pool1_size, pool1_stride)

    # conv2
    if form.get("use_conv2") and form.get("conv2_filters"):
        current_channels = int(form.get("conv2_filters", current_channels))

    dense_input_size  = current_channels * current_size * current_size
    dense_output_size = int(form.get("dense_units", 128))

    return {
        "conv1_in_channels": in_c,
        "conv2_in_channels": current_channels,
        "fc_input_size":     dense_input_size,
        "fc_output_size":    dense_output_size,
    }