import logging
# decorator that logs errors when exceptions occur within any function

# Create a logger
logger = logging.getLogger(__name__)
logging.basicConfig(filename="UCLA_admission.log", level=logging.INFO, format="[%(asctime)s]  %(levelname)s -- %(message)s")

def logging_decorator(func):
    
    def wrapper(*args, **kwargs):
        try:
            # log function called infomation
            logger.info(f"Function '{func.__name__}' started.")
            return func(*args, **kwargs)
        # log errors
        except FileNotFoundError:
            logger.error(f"Cannot find the file in function '{func.__name__}'")
            return None
        except PermissionError:
            logger.error(f"No permission for the file in function '{func.__name__}'")
            return None
        except ValueError:
            logger.error(f"Value Error occured in function '{func.__name__}'")
            return None
        except TypeError:
            logger.error(f"Type Error occured in function '{func.__name__}'")
        except Exception as e:
            logger.error(f"Unexpected error occurred in function '{func.__name__}' with arguments {args} and {kwargs}. Error: {e}")
            return None
        finally:
            # log the end of functions execution
            logger.info(f"Function '{func.__name__}' ended.")
    
    return wrapper