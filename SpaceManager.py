class SpaceManager: 
    """
        Space Manager keeps track of spaces
    """

    def __init__(self):
        self.spaces = {}
    
    def get_all_spaces(self):
        return self.spaces
    
    def get_space(self, space_key):
        return self.spaces.get(space_key, None)
    
    def add_space(self, space):
        self.spaces[space.key] = space