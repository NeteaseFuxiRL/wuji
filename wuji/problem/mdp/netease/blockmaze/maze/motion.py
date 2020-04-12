from collections import namedtuple


VonNeumannMotion = namedtuple('VonNeumannMotion', 
                              ['north', 'south', 'west', 'east'] )
VonNeumannMotion.__new__.__defaults__=([-1, 0], [1, 0], [0, -1], [0, 1])


MooreMotion = namedtuple('MooreMotion', 
                         ['north', 'south', 'west', 'east', 
                          'northwest', 'northeast', 'southwest', 'southeast'])
MooreMotion.__new__.__defaults__ = ([-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1])
