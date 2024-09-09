from class_robots import Coppelia, P3DX, state
from time import time

def avoid(metrics, status):
    if metrics["readings"][1] > 0.8 and metrics["readings"][2] > 0.8 and metrics["readings"][3] > 0.8 and metrics["readings"][4] > 0.8 and metrics["readings"][5] > 0.8 and metrics["readings"][6] > 0.8:
        status.wall_find = False
    if metrics["front_metrics"] == 1.0 and status.right_turn == True and  metrics["left_metrics"] < 0.1 and metrics["right_metrics"] < 0.1:
        ispeed, rspeed = move_forward()
    elif metrics["front_metrics"] < 0.5:
        ispeed, rspeed = turn_left(1)
        status.wall_find = True
        status.right_turn = False
    elif metrics["left_metrics"] < 0.5 and metrics["right_metrics"] < 0.5 and status.wall_find == True:
        middle_metric = metrics["left_metrics"] / metrics["right_metrics"]
        ispeed, rspeed = zigzag(middle_metric)
    elif metrics["right_metrics"] < 0.5 and status.wall_find == True:
        ispeed, rspeed = turn_left(metrics["right_metrics"])
    elif metrics["left_metrics"] < 0.5 and status.wall_find == True:
        ispeed, rspeed = turn_right(metrics["left_metrics"])
    elif metrics["right_metrics"] > 0.3 and status.wall_find == True:
        ispeed, rspeed = turn_right(metrics["right_metrics"])
    else:
        ispeed, rspeed = move_forward()
    return ispeed, rspeed, status

def follow_right_wall(metrics):
    if metrics["right_wall_metrics"] > 0.5:
        ispeed, rspeed = turn_left(1)
    else:
        ispeed, rspeed = move_forward()
    return ispeed, rspeed

def move_forward():
    ispeed, rspeed = 1, 1
    return ispeed, rspeed

def turn_left(grade):
    ispeed, rspeed = 0, grade
    return ispeed, rspeed

def turn_right(grade):
    ispeed, rspeed = grade, 0
    return ispeed, rspeed

def zigzag(grade):
    ispeed, rspeed = grade, 1
    return (ispeed / max(ispeed, rspeed)), (rspeed / max(ispeed, rspeed))

def get_metrics(readings):
    metrics = {}
    metrics["forward_metrics"] = readings[3] * readings[4]
    metrics["left_metrics"] = readings[1] * readings[2]
    metrics["right_metrics"] = readings[5] * readings[6]
    metrics["right_wall_metrics"] = readings[7]
    metrics["left_wall_metrics"] = readings[0]
    metrics["readings"] = readings
    return metrics


def main():
    print('Running...')
    status = state()
    coppelia = Coppelia()
    id = coppelia.sim.getObject('/PioneerP3DX')
    robot = P3DX(coppelia.sim, "PioneerP3DX")
    coppelia.start_simulation()
    print('Simulation started')
    while coppelia.is_running():
        readings = robot.get_sonar()
        print(readings)
        metrics = get_metrics(readings)
        ispeed, rspeed, new_status = avoid(metrics, status)
        print(ispeed, rspeed)
        robot.set_speed(ispeed, rspeed)
        status = new_status
    coppelia.stop_simulation()

if __name__ == '__main__':
    main()