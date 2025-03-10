from .cub import name_list as cub_nl
from .cub import temp as cub_tp

from .car import name_list as car_nl
from .car import temp as car_tp

from .air import name_list as air_nl
from .air import temp as air_tp

from .dog import name_list as dog_nl
from .dog import temp as dog_tp

from .nabird import name_list as nabird_nl
from .nabird import temp as nabird_tp

name_dict = dict()
name_dict['cub'] = (cub_nl, cub_tp)
name_dict['car'] = (car_nl, car_tp)
name_dict['air'] = (air_nl, air_tp)
name_dict['dog'] = (dog_nl, dog_tp)
name_dict['nabird'] = (nabird_nl, nabird_tp)
