
## Sensor Settings
class LiDAR:
    def __init__(self, name, position, area_size, lidar_radius, mapinfo):
        self.name = name
        self.ogm = OccupancyGridMap(np.ones((area_size[0], area_size[1])), 1)
        self.lidar_radius = lidar_radius
        self.position = position
        self.ogm.mapinfo = mapinfo
        self.count_people = {}

    def getCount(self):
        return self.count_people

    def scan(self, agent_list):
        def _isIntersect(a, b, c, d):
            tc = (a[0] - b[0]) * (c[1] - a[1]) - (a[1] - b[1]) * (c[0] - a[0])
            td = (a[0] - b[0]) * (d[1] - a[1]) - (a[1] - b[1]) * (d[0] - a[0])
            if (tc * td > 0):
                return False

            tc = (c[0] - d[0]) * (a[1] - c[1]) - (c[1] - d[1]) * (a[0] - c[0])
            td = (c[0] - d[0]) * (b[1] - c[1]) - (c[1] - d[1]) * (b[0] - c[0])
            if (tc * td > 0):
                return False
            return True

        def _line_cross_point(P0, P1, Q0, Q1):
            x0, y0 = P0
            x1, y1 = P1
            x2, y2 = Q0
            x3, y3 = Q1
            a0 = x1 - x0
            b0 = y1 - y0
            a2 = x3 - x2
            b2 = y3 - y2

            d = a0*b2 - a2*b0
            if d == 0:
                # two lines are parallel
                return None

            # s = sn/d
            sn = b2 * (x2-x0) - a2 * (y2-y0)
            # t = tn/d
            # tn = b0 * (x2-x0) - a0 * (y2-y0)
            return int(Decimal(str(x0 + a0*sn/d)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)), int(Decimal(str(y0 + b0*sn/d)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))

        def makeWall(robot_position, lidar_radius):
            _wall = []
            left_upper = [robot_position[0] - lidar_radius,
                          robot_position[1] + lidar_radius]
            left_lower = [robot_position[0] - lidar_radius,
                          robot_position[1] - lidar_radius]
            right_upper = [robot_position[0] + lidar_radius,
                           robot_position[1] + lidar_radius]
            right_lower = [robot_position[0] + lidar_radius,
                           robot_position[1] - lidar_radius]
            _wall.extend([[left_lower[0], left_lower[1] + i]
                         for i in range(2*lidar_radius)])
            _wall.extend([[left_upper[0] + i, left_upper[1]]
                         for i in range(2*lidar_radius)])
            _wall.extend([[right_upper[0], right_upper[1] - i]
                         for i in range(2*lidar_radius)])
            _wall.extend([[right_lower[0] - i, right_lower[1]]
                         for i in range(2*lidar_radius)])
            return _wall

        def _one_azimuth_scan(ogm_info, mapinfo, robot_position, target_point, agent_list, lidar_radius):
            ogm_data = np.ones((ogm_info["size"][0], ogm_info["size"][1]))
            ogm = OccupancyGridMap(ogm_data, 1)
            tm = TimeMeasure(3)
            tm.setCPName(1, "calculate cross point")
            tm.setCPName(2, "find occulusion agent")
            tm.setCPName(3, "set data")
            # Find intersection point with walls
            occulusion_agent_list = []
            cross_map_wall_point = target_point
            tm.checkpoint(1)
            for idx_mapinfo in range(1, len(mapinfo)):
                wall_start = mapinfo[idx_mapinfo - 1]
                wall_end = mapinfo[idx_mapinfo]
                is_intersect = _isIntersect(
                    robot_position, target_point, wall_start, wall_end)
                # 交点と人の位置関係を考えないとおかしいことになる
                if (is_intersect and target_point[0] > 0 and target_point[1] > 0):
                    try:
                        _cross_map_wall_point = _line_cross_point(
                            robot_position, target_point, wall_start, wall_end)
                        if (_cross_map_wall_point is None):
                            continue
                        elif (cross_map_wall_point == target_point or calcEuclidean(robot_position, _cross_map_wall_point) < calcEuclidean(robot_position, cross_map_wall_point)):
                            cross_map_wall_point = _cross_map_wall_point
                    except:
                        # print("robot_pos : {}".format(robot_position))
                        # print("target_pos : {}".format(target_point))
                        # print("first_point : {}".format(first_point))
                        # print("second_point : {}".format(second_point))
                        # print("mapinfo : {}".format(mapinfo))
                        pass
            tm.checkpoint(1)

            # Find occluding agent
            tm.checkpoint(2)
            # print("start")
            laser_beams = lg.bresenham(robot_position, cross_map_wall_point)
            # print("end")
            occulusion_agent = [10000, None]
            isOcculusion_laser_beams = [False for i in range(len(laser_beams))]
            for agent in agent_list:
                # if (not agent.isStarted or agent.isArrived):
                if (not agent[1] or agent[0]):
                    continue
                checkOcculusion = [calcEuclidean(agent[2], laser_beam) < (
                    agent[3]) for laser_beam in laser_beams]
                if any(checkOcculusion):
                    isOcculusion_laser_beams = [x | y for x, y in zip(
                        isOcculusion_laser_beams, checkOcculusion)]
                    occ_idx = checkOcculusion.index(True)
                    if (occ_idx < occulusion_agent[0] and calcEuclidean(agent[2], robot_position) < lidar_radius):
                        occulusion_agent[0] = occ_idx
                        occulusion_agent[1] = agent[4]

            laser_point_list = laser_beams
            if any(isOcculusion_laser_beams):
                lb_idx = isOcculusion_laser_beams.index(True)
                laser_point_list = laser_beams[:lb_idx]

                if occulusion_agent[1] is not None and occulusion_agent[1] not in occulusion_agent_list:
                    occulusion_agent_list.append(occulusion_agent[1])
            tm.checkpoint(2)
            tm.checkpoint(3)
            for laser_beam in laser_point_list:
                if (calcEuclidean(laser_beam, robot_position) > lidar_radius or np.all(laser_beam == robot_position)):
                    continue
                # if (laser_beam.tolist() in self.ogm.mapinfo):
                #     break
                try:
                    ogm.set_data(laser_beam, 0)
                except:
                    pass

            tm.checkpoint(3)
            # tm.result()
            return ogm.data, occulusion_agent_list, tm.getTime()
        tm = TimeMeasure(3)
        tm.checkpoint(1)
        self.ogm.reset_data()
        _wall = makeWall(self.position, self.lidar_radius)
        human_unique_num_dict = {
            agent.unique_num: agent for agent in agent_list}
        tm.checkpoint(1)
        ogm_info = {"size": self.ogm.dim_cells}
        tm.checkpoint(2)
        agent_value_list = [[agent.isArrived, agent.isStarted,
                             agent.position, agent.personalspace, agent.unique_num] for agent in agent_list]
        results = joblib.Parallel(n_jobs=-1, verbose=0, backend='threading')(joblib.delayed(_one_azimuth_scan)(
            ogm_info, self.ogm.mapinfo, self.position, point, agent_value_list, self.lidar_radius) for point in _wall)
        tm.checkpoint(2)
        tm.checkpoint(3)
        time = np.array([0., 0., 0.])
        for result in results:
            self.ogm.data *= result[0]
            for _occulusion_agent in result[1]:
                if (_occulusion_agent not in self.count_people):
                    self.count_people[_occulusion_agent] = human_unique_num_dict[_occulusion_agent]
            time += result[2]
        # print(result[0])
        tm.checkpoint(3)
        # tm.result()
        return self.ogm.data, list(self.count_people.values())