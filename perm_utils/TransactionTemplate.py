class TransactionTemplate: 

    def execute(self, callback):
        try: 
            result = callback()
            return result
        except Exception as e: 
            print(f"Transaction failed: {e}")
            return None