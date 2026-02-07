from views.power_test import PowerTestView

class CognitiveView(PowerTestView):
    def __init__(self, parent):
        super().__init__(parent, title="COGNITIVE")
