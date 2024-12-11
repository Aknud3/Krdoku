class App:
    """Main class for the application.
    """
    def __init__(self, main_menu):
        """ Initialize the application.
        """
        self.self = self
        self.main_menu = main_menu
             
    def run(self):
        """ Run function for the app.
        """
        self.main_menu.run()
