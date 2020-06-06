class UnmannedVehicle(object):
    """ This class implements a unmanned vehicle instance
    to use as node in a graph. """
    def __init__(self, id, *args, **kwargs):
        self.id = id
        self.type = kwargs.get('type', 'UnmannedVehicle')

    def __str__(self):
        return '{}({})'.format(self.type, self.id)