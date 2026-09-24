import math


class BilliardScoring:
    def __init__(self):
        # --- 比赛状态 ---
        self.score_solid = 0
        self.score_stripe = 0
        self.foul_flag = False
        self.game_over = False
        self.winner = None

        # --- 空间配置 ---
        # ⚠️ 用你 get_points.py 点出来的 6 个袋口中心

        self.pockets = [(87, 29), (647, 19), (1210, 28), (1209, 689), (647, 692), (87, 686)]
        # UI 画圈半径（仅显示）
        self.pocket_radius = 45

        # --- 进球判定参数 ---
        self.goal_radius_standard = 85
        self.goal_radius_black8 = 140   # 黑八更宽容（你这里可继续微调）

        # 消失确认帧数（普通球 / 黑八分开）
        self.confirm_missing_frames = 10
        self.confirm_missing_frames_black8 = 4   # ✅ 黑八更快确认，避免错过

        # 袋口冷却（防重复加分）
        self.pocket_cooldown_frames = 30
        self.pocket_cooldowns = {i: 0 for i in range(6)}

        # --- 轨迹参数（增强版）---
        self.history_len = 8                   # 轨迹长度稍微加长
        self.fast_speed_px = 12.0
        self.max_dynamic_bonus = 40
        self.near_margin = 24
        self.min_cos_toward_pocket = 0.08      # 稍微放宽一点（更容易判“朝袋口”）

        # --- 类别映射参数 ---
        # 你的数据集如果黑八不是 14，就改这里
        self.black8_class_id = 14

        # --- 内部状态 ---
        self.frame_count = 0
        self.active_balls = {}   # tid -> {'pos','cls','history'}
        self.missing_balls = {}  # tid -> {'pos','cls','history','missing'}

    # -------------------- 类别映射 --------------------
    def _get_ball_type(self, class_id):
        cid = int(class_id)

        # 白球
        if cid == 0:
            return 'white'

        # 黑八（可配置）
        if cid == self.black8_class_id:
            return 'black8'

        # 保持你原来的映射（按你之前的类别顺序）
        if cid in [1, 8, 9, 10, 11, 12, 13]:
            return 'solid'
        if cid in [2, 3, 4, 5, 6, 7, 15]:
            return 'stripe'

        return 'unknown'

    # -------------------- 几何工具 --------------------
    def _dist(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _point_to_segment_distance(self, p, a, b):
        px, py = p
        ax, ay = a
        bx, by = b

        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay

        ab2 = abx * abx + aby * aby
        if ab2 == 0:
            return math.hypot(px - ax, py - ay)

        t = (apx * abx + apy * aby) / ab2
        t = max(0.0, min(1.0, t))
        cx = ax + t * abx
        cy = ay + t * aby
        return math.hypot(px - cx, py - cy)

    def _segment_min_dist_to_pocket(self, history, pocket):
        if not history:
            return 9999.0
        if len(history) == 1:
            return self._dist(history[0], pocket)

        best = 9999.0
        for i in range(1, len(history)):
            d = self._point_to_segment_distance(pocket, history[i - 1], history[i])
            if d < best:
                best = d
        return best

    def _velocity_info(self, history, pocket):
        if len(history) < 2:
            return 0.0, 0.0, 0.0

        p_prev = history[-2]
        p_last = history[-1]

        vx = p_last[0] - p_prev[0]
        vy = p_last[1] - p_prev[1]
        speed = math.hypot(vx, vy)

        d_prev = self._dist(p_prev, pocket)
        d_last = self._dist(p_last, pocket)
        approach_delta = d_prev - d_last  # >0 表示靠近袋口

        tx = pocket[0] - p_prev[0]
        ty = pocket[1] - p_prev[1]
        tnorm = math.hypot(tx, ty)

        if speed <= 1e-6 or tnorm <= 1e-6:
            cos_toward = 0.0
        else:
            cos_toward = (vx * tx + vy * ty) / (speed * tnorm)

        return speed, approach_delta, cos_toward

    def _pick_best_pocket(self, history):
        """综合最后点+轨迹线段，选最可能袋口"""
        if not history:
            return -1, 9999.0, 9999.0

        last = history[-1]
        best_pidx = -1
        best_score = 1e9
        best_last_dist = 9999.0
        best_seg_dist = 9999.0

        for i, pocket in enumerate(self.pockets):
            d_last = self._dist(last, pocket)
            d_seg = self._segment_min_dist_to_pocket(history, pocket)

            # 轨迹线段更重要（高速球、黑八最后一杆很关键）
            score = d_seg * 0.75 + d_last * 0.25
            if score < best_score:
                best_score = score
                best_pidx = i
                best_last_dist = d_last
                best_seg_dist = d_seg

        return best_pidx, best_last_dist, best_seg_dist

    # -------------------- 进球判定核心 --------------------
    def _is_valid_pot(self, ball_type, history, pocket):
        base_threshold = self.goal_radius_black8 if ball_type == 'black8' else self.goal_radius_standard

        last = history[-1]
        last_dist = self._dist(last, pocket)
        seg_min_dist = self._segment_min_dist_to_pocket(history, pocket)
        speed, approach_delta, cos_toward = self._velocity_info(history, pocket)

        # 高速球动态放宽
        dynamic_bonus = min(self.max_dynamic_bonus, max(0.0, (speed - self.fast_speed_px) * 2.0))
        dyn_threshold = base_threshold + dynamic_bonus

        # 近袋证据
        near_r = base_threshold + self.near_margin
        near_count = sum(1 for p in history if self._dist(p, pocket) <= near_r)

        # 普通条件
        cond_normal = (
            last_dist <= dyn_threshold and
            (
                approach_delta > -2.5 or
                near_count >= 2 or
                ball_type == 'black8'
            ) and
            (
                cos_toward >= self.min_cos_toward_pocket or
                speed < 3.0 or
                near_count >= 3 or
                ball_type == 'black8'
            )
        )

        # 高速球条件（轨迹穿袋）
        cond_fast = (
            speed >= self.fast_speed_px and
            seg_min_dist <= dyn_threshold and
            (
                approach_delta > 0.5 or
                cos_toward > 0.0 or
                ball_type == 'black8'
            )
        )

        # ✅ 黑八专用兜底（更强）
        # 只要黑八轨迹非常贴近袋口，或者最后点+近袋证据成立，就判
        cond_black8_special = False
        if ball_type == 'black8':
            cond_black8_special = (
                seg_min_dist <= (self.goal_radius_black8 + 30) or
                (last_dist <= (self.goal_radius_black8 + 20) and near_count >= 1) or
                (near_count >= 2 and (approach_delta > -4.0 or cos_toward > -0.2))
            )

        is_goal = cond_normal or cond_fast or cond_black8_special

        debug_msg = (
            f"[last={last_dist:.1f}, seg={seg_min_dist:.1f}, speed={speed:.1f}, "
            f"approach={approach_delta:.1f}, cos={cos_toward:.2f}, near={near_count}, "
            f"th={dyn_threshold:.1f}]"
        )
        return is_goal, debug_msg

    def _missing_confirm_frames_for_ball(self, cls_id):
        """黑八单独更快确认，减少漏判"""
        return self.confirm_missing_frames_black8 if self._get_ball_type(cls_id) == 'black8' else self.confirm_missing_frames

    # -------------------- 主更新逻辑 --------------------
    def update(self, tracks):
        if self.game_over:
            return

        self.frame_count += 1
        current_ids = set()

        # 1) 解析当前帧
        if tracks.boxes.id is not None:
            boxes = tracks.boxes.xyxy.cpu().numpy()
            ids = tracks.boxes.id.cpu().numpy()
            clss = tracks.boxes.cls.cpu().numpy()

            for box, tid, cls_id in zip(boxes, ids, clss):
                tid = int(tid)
                current_ids.add(tid)

                cx = float((box[0] + box[2]) / 2)
                cy = float((box[1] + box[3]) / 2)
                pos = (cx, cy)

                # 如果之前在 missing（遮挡/抖动）里，取回旧轨迹
                if tid in self.missing_balls:
                    old_hist = self.missing_balls[tid].get('history', [])
                    del self.missing_balls[tid]
                else:
                    old_hist = self.active_balls.get(tid, {}).get('history', [])

                history = old_hist + [pos]
                if len(history) > self.history_len:
                    history = history[-self.history_len:]

                self.active_balls[tid] = {
                    'pos': pos,
                    'cls': cls_id,
                    'history': history
                }

        # 2) 本帧消失的球 -> missing
        for tid in list(self.active_balls.keys()):
            if tid not in current_ids:
                info = self.active_balls[tid]
                self.missing_balls[tid] = {
                    'pos': info['pos'],
                    'cls': info['cls'],
                    'history': info.get('history', [info['pos']]),
                    'missing': 0
                }
                del self.active_balls[tid]

        # 3) missing 区延迟判定
        for tid in list(self.missing_balls.keys()):
            info = self.missing_balls[tid]
            info['missing'] += 1

            cls_id = info['cls']
            ball_type = self._get_ball_type(cls_id)
            need_confirm = self._missing_confirm_frames_for_ball(cls_id)

            if info['missing'] == need_confirm:
                history = info.get('history', [info['pos']])

                # 选最可能袋口
                best_pidx, last_dist, seg_dist = self._pick_best_pocket(history)
                if best_pidx != -1:
                    pocket = self.pockets[best_pidx]
                    is_goal, dbg = self._is_valid_pot(ball_type, history, pocket)

                    if is_goal:
                        # 冷却防抖
                        if self.frame_count - self.pocket_cooldowns[best_pidx] > self.pocket_cooldown_frames:
                            self._trigger_score_event(ball_type)
                            self.pocket_cooldowns[best_pidx] = self.frame_count
                            print(
                                f"🎯 进球判定成功 [类型:{ball_type}] [袋口:{best_pidx}] "
                                f"[last:{last_dist:.1f}] [seg:{seg_dist:.1f}] {dbg}"
                            )
                        else:
                            print(f"🛡️ 冷却中，忽略重复进球 [类型:{ball_type}] [袋口:{best_pidx}] {dbg}")
                    else:
                        print(
                            f"❌ 未判定进球 [类型:{ball_type}] [袋口:{best_pidx}] "
                            f"[last:{last_dist:.1f}] [seg:{seg_dist:.1f}] {dbg}"
                        )

                del self.missing_balls[tid]

            # 安全清理
            elif info['missing'] > max(self.confirm_missing_frames, self.confirm_missing_frames_black8) + 30:
                del self.missing_balls[tid]

    # -------------------- 事件处理 --------------------
    def _trigger_score_event(self, ball_type):
        if ball_type == 'solid':
            self.score_solid += 1
        elif ball_type == 'stripe':
            self.score_stripe += 1
        elif ball_type == 'white':
            self.foul_flag = True
            print("⚠️ 犯规：母球入袋！")
        elif ball_type == 'black8':
            self.game_over = True
            self.winner = "Black 8 (Finish)"
            print("🏁 比赛结束：黑八入袋！")

    # -------------------- 对外接口 --------------------
    def get_status(self):
        return {
            "solid": self.score_solid,
            "stripe": self.score_stripe,
            "foul": self.foul_flag,
            "game_over": self.game_over,
            "winner": self.winner
        }

    def reset_foul(self):
        self.foul_flag = False

    def reset_game(self):
        self.score_solid = 0
        self.score_stripe = 0
        self.foul_flag = False
        self.game_over = False
        self.winner = None
        self.frame_count = 0
        self.active_balls.clear()
        self.missing_balls.clear()
        self.pocket_cooldowns = {i: 0 for i in range(6)}
        print("🔄 数据已完全清零重置")