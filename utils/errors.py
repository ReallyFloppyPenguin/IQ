class UnevenDataErr(SyntaxError):
    """Error for when the data is not even"""
    def __init__(self, *args: object) -> None:
        
        super().__init__(*args)