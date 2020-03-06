#!/usr/bin/env python3
import unittest
import importlib
from selfdrive.car.fingerprints import all_known_cars
from selfdrive.car.car_helpers import interfaces
from selfdrive.car.fingerprints import _FINGERPRINTS as FINGERPRINTS

from cereal import car


class TestCarParam(unittest.TestCase):
  def test_creating_car_params(self):
    all_cars = all_known_cars()

    for car_name in all_cars:
      fingerprint = FINGERPRINTS[car_name][0]

      CarInterface, CarController, CarState = interfaces[car_name]
      fingerprints = {
        0: fingerprint,
        1: fingerprint,
        2: fingerprint,
      }

      car_fw = []

      for has_relay in [True, False]:
        car_params = CarInterface.get_params(car_name, fingerprints, has_relay, car_fw)
        car_interface, _ = CarInterface(car_params, CarController, CarState), car_params
        assert car_params
        assert car_interface

        # Run car interface once
        CC = car.CarControl.new_message()
        car_interface.update(CC, [])

      # Test radar interface
      RadarInterface = importlib.import_module('selfdrive.car.%s.radar_interface' % car_params.carName).RadarInterface
      radar_interface = RadarInterface(car_params)
      assert radar_interface

      # Run car interface once
      radar_interface.update([])
      if hasattr(radar_interface, '_update') and hasattr(radar_interface, 'trigger_msg'):
        radar_interface._update([radar_interface.trigger_msg])

if __name__ == "__main__":
  unittest.main()
