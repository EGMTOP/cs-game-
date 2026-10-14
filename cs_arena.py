"""
CS ARENA  -  original Counter-Strike-style bomb defusal FPS in Pygame
=====================================================================
Pseudo-3D (raycasting) engine, fully vectorised with NumPy so it stays smooth.
You are the Counter-Terrorist. Bots are Terrorists who try to plant the bomb.

Install:   pip install pygame numpy
Run:       python cs_arena.py
Build EXE: pip install pyinstaller
           pyinstaller --onefile --noconsole --name CSArena cs_arena.py
           (the .exe appears in the "dist" folder)

CONTROLS
  WASD move | Mouse look | LMB shoot | RMB scope (sniper) | R reload
  Shift walk (quiet, accurate) | E hold to defuse | 1-4 buy / switch weapon
  Q last weapon | [ ] sensitivity | F11 fullscreen | Esc menu/quit
"""
import math
import random
import sys
from collections import deque

import numpy as np
import pygame

# ----------------------------------------------------------------- config
WIN_W, WIN_H = 1280, 720          # window size
RW, RH = 480, 270                 # internal 3D resolution (scaled up)
PLANE = 0.9                       # camera plane length (FOV ~ 84 deg)
FPS_CAP = 144
MAXP = 70                         # max pitch in internal pixels
BUY_TIME, ROUND_TIME, BOMB_TIME = 12.0, 100.0, 35.0
PLANT_TIME, DEFUSE_TIME = 3.0, 5.0
WIN_ROUNDS = 7

# 28 x 20 map.  # wall | . floor | T bot spawn | C player spawn | A,B bomb sites
MAP = [
    "############################",
    "#T.T...#..........#......A.#",
    "#......#..........#........#",
    "#.T.T..#...####...#...##...#",
    "#..........#..#.......##...#",
    "#..........#..#...#........#",
    "#####.######..#...####.#####",
    "#..........................#",
    "#...###......####......###.#",
    "#...#..........#.......#...#",
    "#...#..###.....#...##..#...#",
    "#......#.#.........##......#",
    "#......###......#..........#",
    "####.#####..#...#...######.#",
    "#B.........#...............#",
    "#..........#..###..........#",
    "#..####....#..#.#....####..#",
    "#..#..#........#.....#.....#",
    "#.....#..............#...C.#",
    "############################",
]
MH, MW = len(MAP), len(MAP[0])
GRID = np.zeros((MH, MW), dtype=np.int8)
T_SPAWNS, SITES, C_SPAWN = [], {}, (MW - 3.5, MH - 2.5)
for _y, _row in enumerate(MAP):
    assert len(_row) == MW, f"map row {_y} has wrong length"
    for _x, _c in enumerate(_row):
        if _c == "#":
            GRID[_y, _x] = 1
        elif _c == "T":
            T_SPAWNS.append((_x + .5, _y + .5))
        elif _c == "C":
            C_SPAWN = (_x + .5, _y + .5)
        elif _c in "AB":
            SITES[_c] = (_x + .5, _y + .5)
TILE = ((np.arange(MW)[None, :] * 7 + np.arange(MH)[:, None] * 3) % 3).astype(np.int8)
WALL_COL = np.array([[150, 110, 80], [120, 125, 135], [135, 95, 70]], dtype=np.float32)

WEAPONS = {
    "Pistol": dict(price=0,    dmg=26,  rate=0.17, mag=12, res=48, spread=0.010, kick=3.0,  auto=False, reload=1.4),
    "SMG":    dict(price=1200, dmg=19,  rate=0.075, mag=30, res=90, spread=0.026, kick=1.6, auto=True,  reload=1.9),
    "Rifle":  dict(price=2700, dmg=32,  rate=0.10, mag=30, res=90, spread=0.018, kick=2.4,  auto=True,  reload=2.3),
    "Sniper": dict(price=4700, dmg=115, rate=1.30, mag=5,  res=20, spread=0.002, kick=9.0,  auto=False, reload=3.0),
}
ORDER = ["Pistol", "SMG", "Rifle", "Sniper"]
DIFF = {  # reaction time, base accuracy, damage, speed
    "EASY":   dict(react=0.70, acc=0.35, dmg=8,  speed=1.9),
    "NORMAL": dict(react=0.45, acc=0.50, dmg=11, speed=2.3),
    "HARD":   dict(react=0.25, acc=0.70, dmg=14, speed=2.7),
}


# ----------------------------------------------------------------- helpers
def walkable(x, y):
    xi, yi = int(x), int(y)
    return 0 <= xi < MW and 0 <= yi < MH and GRID[yi, xi] == 0


def can_stand(x, y, r=0.22):
    return all(walkable(x + dx, y + dy) for dx in (-r, r) for dy in (-r, r))


def move_slide(x, y, dx, dy):
    if can_stand(x + dx, y):
        x += dx
    if can_stand(x, y + dy):
        y += dy
    return x, y


def line_of_sight(x0, y0, x1, y1):
    d = math.hypot(x1 - x0, y1 - y0)
    n = max(1, int(d / 0.15))
    for i in range(1, n):
        t = i / n
        if not walkable(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t):
            return False
    return True


def bfs(start, goal):
    """4-way grid pathfinding; returns list of cells from start to goal."""
    if start == goal:
        return [start]
    prev = {start: None}
    q = deque([start])
    while q:
        c = q.popleft()
        if c == goal:
            break
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + dx, c[1] + dy)
            if n not in prev and 0 <= n[0] < MW and 0 <= n[1] < MH and GRID[n[1], n[0]] == 0:
                prev[n] = c
                q.append(n)
    if goal not in prev:
        return [start]
    path, c = [], goal
    while c is not None:
        path.append(c)
        c = prev[c]
    return path[::-1]


def make_background():
    """Ceiling + floor gradient, taller than the screen so pitch can slide it."""
    h = RH + 2 * MAXP
    ys = np.arange(h, dtype=np.float32)
    horizon = h / 2
    bg = np.zeros((h, 3), dtype=np.float32)
    up = ys < horizon
    k = (horizon - ys[up]) / horizon
    bg[up] = np.stack([40 + 50 * k, 50 + 60 * k, 70 + 90 * k], axis=1)
    dn = ~up
    k = (ys[dn] - horizon) / horizon
    bg[dn] = np.stack([30 + 55 * k, 28 + 50 * k, 26 + 42 * k], axis=1)
    return np.repeat(bg[None, :, :], RW, axis=0)          # (RW, h, 3)


BG = make_background().astype(np.float32)


def cast_rays(px, py, dx, dy, plx, ply):
    """Vectorised DDA for every screen column. Returns perp dist, side, wall-x, tile."""
    cam = np.linspace(-1, 1, RW, dtype=np.float64)
    rdx, rdy = dx + plx * cam, dy + ply * cam
    rdx = np.where(np.abs(rdx) < 1e-9, 1e-9, rdx)
    rdy = np.where(np.abs(rdy) < 1e-9, 1e-9, rdy)
    ddx, ddy = np.abs(1 / rdx), np.abs(1 / rdy)
    mx = np.full(RW, int(px))
    my = np.full(RW, int(py))
    stx = np.where(rdx < 0, -1, 1)
    sty = np.where(rdy < 0, -1, 1)
    sdx = np.where(rdx < 0, (px - mx) * ddx, (mx + 1 - px) * ddx)
    sdy = np.where(rdy < 0, (py - my) * ddy, (my + 1 - py) * ddy)
    side = np.zeros(RW, dtype=np.int8)
    hit = np.zeros(RW, dtype=bool)
    for _ in range(60):
        step_x = sdx < sdy
        act = ~hit
        mvx, mvy = act & step_x, act & ~step_x
        sdx = np.where(mvx, sdx + ddx, sdx)
        mx = np.where(mvx, mx + stx, mx)
        sdy = np.where(mvy, sdy + ddy, sdy)
        my = np.where(mvy, my + sty, my)
        side = np.where(act, np.where(step_x, 0, 1), side).astype(np.int8)
        hit |= GRID[my, mx] > 0
        if hit.all():
            break
    perp = np.where(side == 0, sdx - ddx, sdy - ddy)
    perp = np.maximum(perp, 0.05)
    wallx = np.where(side == 0, py + perp * rdy, px + perp * rdx)
    wallx -= np.floor(wallx)
    return perp, side, wallx, TILE[my, mx]


# ----------------------------------------------------------------- sound
def make_sounds():
    snd = {}
    try:
        pygame.mixer.init(44100, -16, 1, 512)

        def mk(freq, dur, noise=0.0, decay=8.0, vol=0.5):
            t = np.linspace(0, dur, int(44100 * dur), False)
            wave = np.sin(2 * np.pi * freq * t) * (1 - noise) + np.random.uniform(-1, 1, t.size) * noise
            wave *= np.exp(-decay * t) * vol
            return pygame.sndarray.make_sound((wave * 32767).astype(np.int16))

        snd["Pistol"] = mk(180, 0.18, 0.8, 22)
        snd["SMG"] = mk(220, 0.10, 0.8, 30, 0.4)
        snd["Rifle"] = mk(140, 0.16, 0.85, 20, 0.55)
        snd["Sniper"] = mk(70, 0.45, 0.8, 8, 0.8)
        snd["hit"] = mk(900, 0.05, 0.0, 40, 0.35)
        snd["hurt"] = mk(120, 0.15, 0.4, 14, 0.6)
        snd["beep"] = mk(1500, 0.08, 0.0, 20, 0.3)
        snd["boom"] = mk(45, 1.2, 0.7, 3, 0.9)
        snd["reload"] = mk(500, 0.06, 0.5, 30, 0.3)
    except Exception:
        pass
    return snd


# ----------------------------------------------------------------- game
class Bot:
    def __init__(self, pos, carrier=False):
        self.x, self.y = pos
        self.hp = 100
        self.carrier = carrier
        self.react = 0.0
        self.cd = random.uniform(0.2, 0.6)
        self.path, self.path_t = [], 0.0
        self.plant = 0.0
        self.flash = 0.0
        self.strafe = random.choice((-1, 1))
        self.strafe_t = 0.0
        self.target_site = random.choice(list(SITES))
        self.hunt_delay = random.uniform(15, 35)


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("CS ARENA")
        self.fullscreen = False
        self.win = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
        self.small = pygame.Surface((RW, RH))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas,couriernew,monospace", 22, bold=True)
        self.big = pygame.font.SysFont("consolas,couriernew,monospace", 56, bold=True)
        self.tiny = pygame.font.SysFont("consolas,couriernew,monospace", 15, bold=True)
        self.snd = make_sounds()
        self.sens = 0.0022
        self.state = "menu"
        self.diff = "NORMAL"
        self.zbuf = np.full(RW, 99.0)
        self.running = True
        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)

    # ---- match / round setup
    def new_match(self):
        self.ct_score = self.t_score = 0
        self.money = 800
        self.owned = {"Pistol"}
        self.mags = {w: WEAPONS[w]["mag"] for w in WEAPONS}
        self.reserve = {w: WEAPONS[w]["res"] for w in WEAPONS}
        self.weapon = "Pistol"
        self.last_weapon = "Pistol"
        self.round_no = 0
        self.streak_loss = 0
        self.start_round()
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        pygame.mouse.get_rel()

    def start_round(self):
        self.round_no += 1
        self.px, self.py = C_SPAWN
        self.ang = math.pi
        self.pitch = 0.0
        self.hp, self.armor = 100, 0
        self.reload_t = 0.0
        self.cd = 0.0
        self.kick = 0.0
        self.bob = 0.0
        self.scoped = False
        self.hitmark = 0.0
        self.hurt = 0.0
        self.flash_t = 0.0
        self.defuse = 0.0
        self.bomb = None            # (x, y, timer)
        self.killfeed = []
        self.phase = "buy"
        self.timer = BUY_TIME
        self.msg, self.msg_t = "", 0.0
        self.end_t = 0.0
        self.boom = 0.0
        n = len(T_SPAWNS)
        self.bots = [Bot(T_SPAWNS[i % n], carrier=(i == 0)) for i in range(4)]
        for b, off in zip(self.bots, [(0, 0), (0.4, 0), (0, 0.4), (0.4, 0.4)]):
            if can_stand(b.x + off[0], b.y + off[1]):
                b.x += off[0]
                b.y += off[1]

    def end_round(self, ct_won, text):
        if self.phase == "end":
            return
        self.phase = "end"
        self.end_t = 4.0
        self.msg, self.msg_t = text, 4.0
        if ct_won:
            self.ct_score += 1
            self.money = min(16000, self.money + 3000)
            self.streak_loss = 0
        else:
            self.t_score += 1
            self.streak_loss = min(4, self.streak_loss + 1)
            self.money = min(16000, self.money + 1400 + 500 * self.streak_loss)
            self.owned = {"Pistol"}
            self.weapon = "Pistol"
            self.mags = {w: WEAPONS[w]["mag"] for w in WEAPONS}
            self.reserve = {w: WEAPONS[w]["res"] for w in WEAPONS}

    def play(self, name):
        s = self.snd.get(name)
        if s:
            s.play()

    # ---- weapon actions
    def start_reload(self):
        w = WEAPONS[self.weapon]
        if self.reload_t > 0 or self.mags[self.weapon] >= w["mag"] or self.reserve[self.weapon] <= 0:
            return
        self.reload_t = w["reload"]
        self.play("reload")

    def buy(self, name):
        if name in self.owned:
            self.switch(name)
            return
        price = WEAPONS[name]["price"]
        if self.phase == "buy" and self.money >= price:
            self.money -= price
            self.owned.add(name)
            self.mags[name] = WEAPONS[name]["mag"]
            self.reserve[name] = WEAPONS[name]["res"]
            self.switch(name)
            if name != "Pistol":
                self.armor = 100

    def switch(self, name):
        if name in self.owned and name != self.weapon:
            self.last_weapon, self.weapon = self.weapon, name
            self.reload_t, self.scoped = 0.0, False

    def project(self, x, y):
        """Project a world point; returns (screen_x, depth) or None."""
        dx, dy = math.cos(self.ang), math.sin(self.ang)
        plx, ply = -dy * PLANE, dx * PLANE
        rx, ry = x - self.px, y - self.py
        inv = 1.0 / (plx * dy - dx * ply)
        tx = inv * (dy * rx - dx * ry)
        ty = inv * (-ply * rx + plx * ry)
        if ty < 0.15:
            return None
        return RW / 2 * (1 + tx / ty), ty

    def shoot(self):
        w = WEAPONS[self.weapon]
        if self.reload_t > 0 or self.cd > 0 or self.phase == "end":
            return
        if self.mags[self.weapon] <= 0:
            self.start_reload()
            return
        self.mags[self.weapon] -= 1
        self.cd = w["rate"]
        self.play(self.weapon)
        moving = self.speed_now > 0.1
        spread = w["spread"] * (1 + (2.2 if moving else 0) + (0.4 if self.sprint else -0.35)) \
            * (0.2 if self.scoped and self.weapon == "Sniper" and not moving else 1)
        self.kick = min(1.0, self.kick + 0.35)
        self.pitch = min(MAXP, self.pitch + w["kick"])
        self.ang += random.uniform(-1, 1) * w["kick"] * 0.0016
        ox = random.uniform(-1, 1) * spread * RW * 2.2
        oy = random.uniform(-1, 1) * spread * RW * 2.2
        col = int(np.clip(RW // 2 + ox, 0, RW - 1))
        row = RH / 2 + oy
        horizon = RH / 2 + self.pitch
        best = None
        for b in self.bots:
            if b.hp <= 0:
                continue
            p = self.project(b.x, b.y)
            if not p:
                continue
            sx, depth = p
            if depth > self.zbuf[col]:
                continue
            lh = (RW / (2 * PLANE)) / depth
            bottom = horizon + lh / 2
            top = bottom - lh * 0.9
            half = lh * 0.20
            if abs(col - sx) <= half and top <= row <= bottom:
                if best is None or depth < best[1]:
                    best = (b, depth, row < top + lh * 0.9 * 0.22)
        if best:
            b, depth, head = best
            dmg = w["dmg"] * (2.6 if head else 1.0) * (1.0 - min(0.4, depth * 0.012))
            b.hp -= dmg
            b.flash = 0.12
            self.hitmark = 0.18 if not head else 0.3
            self.play("hit")
            if b.hp <= 0:
                self.money = min(16000, self.money + 300)
                self.killfeed.insert(0, [f"You [{self.weapon}] killed BOT{' *HS*' if head else ''}", 4.0])
                if b.carrier:
                    others = [o for o in self.bots if o.hp > 0 and o is not b]
                    if others and not self.bomb:
                        others[0].carrier = True

    # ---- bot AI
    def update_bots(self, dt):
        cfg = DIFF[self.diff]
        alive = [b for b in self.bots if b.hp > 0]
        for b in alive:
            b.flash = max(0, b.flash - dt)
            b.cd -= dt
            dist = math.hypot(self.px - b.x, self.py - b.y)
            sees = self.hp > 0 and dist < 20 and line_of_sight(b.x, b.y, self.px, self.py)
            goal = None
            stop = False
            if self.phase == "buy":
                continue
            if sees:
                b.react += dt
                b.plant = 0.0
                goal = (int(self.px), int(self.py)) if dist > 7 else None
                if b.react >= cfg["react"]:
                    stop = dist < 9
                    if b.cd <= 0:
                        b.cd = random.uniform(0.28, 0.5)
                        chance = max(0.08, min(0.9, cfg["acc"] - dist * 0.022))
                        if random.random() < chance:
                            dmg = cfg["dmg"] * random.uniform(0.8, 1.3)
                            if self.armor > 0:
                                absorbed = min(self.armor, dmg * 0.5)
                                self.armor -= absorbed
                                dmg -= absorbed * 0.5
                            self.hp -= dmg
                            self.hurt = 0.35
                            self.play("hurt")
                b.strafe_t -= dt
                if b.strafe_t <= 0:
                    b.strafe_t = random.uniform(0.4, 1.1)
                    b.strafe *= -1
            else:
                b.react = max(0, b.react - dt * 2)
                site = SITES[b.target_site]
                if self.bomb:
                    goal = (int(self.bomb[0]), int(self.bomb[1])) if dist > 99 else None
                    bx, by = self.bomb[0], self.bomb[1]
                    if math.hypot(bx - b.x, by - b.y) > 4:
                        goal = (int(bx), int(by))
                elif b.carrier:
                    goal = (int(site[0]), int(site[1]))
                    if math.hypot(site[0] - b.x, site[1] - b.y) < 0.8:
                        stop = True
                        b.plant += dt
                        if b.plant >= PLANT_TIME:
                            self.bomb = (b.x, b.y, BOMB_TIME)
                            self.phase = "bomb"
                            self.msg, self.msg_t = "BOMB HAS BEEN PLANTED", 3.0
                            self.play("beep")
                elif self.timer < ROUND_TIME - b.hunt_delay:
                    goal = (int(self.px), int(self.py))
                else:
                    goal = (int(site[0]), int(site[1]))
            if goal and not stop:
                b.path_t -= dt
                if b.path_t <= 0 or not b.path:
                    b.path = bfs((int(b.x), int(b.y)), goal)
                    b.path_t = 0.5
                if len(b.path) > 1:
                    cx, cy = b.path[1][0] + .5, b.path[1][1] + .5
                    d = math.hypot(cx - b.x, cy - b.y)
                    sp = cfg["speed"] * (0.55 if sees else 1.0) * dt
                    if d > 0.01:
                        b.x, b.y = move_slide(b.x, b.y, (cx - b.x) / d * sp, (cy - b.y) / d * sp)
                    if d < 0.2:
                        b.path.pop(0)
            elif sees and stop:       # strafe while shooting
                ax, ay = -(self.py - b.y), (self.px - b.x)
                n = math.hypot(ax, ay) or 1
                b.x, b.y = move_slide(b.x, b.y, ax / n * b.strafe * 1.0 * dt, ay / n * b.strafe * 1.0 * dt)

    # ---- update
    def update(self, dt):
        keys = pygame.key.get_pressed()
        mouse = pygame.mouse.get_pressed()
        mdx, mdy = pygame.mouse.get_rel()
        self.ang += mdx * self.sens * (0.45 if self.scoped else 1.0)
        self.pitch = float(np.clip(self.pitch - mdy * self.sens * 260 * (0.45 if self.scoped else 1.0), -MAXP, MAXP))
        self.sprint = not keys[pygame.K_LSHIFT]    # True = running, Shift = walk
        self.speed_now = 0.0
        w = WEAPONS[self.weapon]
        self.scoped = bool(mouse[2]) and self.weapon == "Sniper"

        if self.hp > 0 and self.phase != "end":
            dx, dy = math.cos(self.ang), math.sin(self.ang)
            fx = (keys[pygame.K_w] - keys[pygame.K_s])
            sx = (keys[pygame.K_d] - keys[pygame.K_a])
            if fx or sx:
                n = math.hypot(fx, sx)
                spd = (4.2 if self.sprint else 2.1) * (0.8 if self.weapon == "Sniper" else 1.0)
                mvx = (dx * fx - dy * sx) / n * spd * dt
                mvy = (dy * fx + dx * sx) / n * spd * dt
                self.px, self.py = move_slide(self.px, self.py, mvx, mvy)
                self.speed_now = spd
                self.bob += dt * spd * 2.2
            if mouse[0] and (w["auto"] or not self._mouse_was):
                self.shoot()
            self._mouse_was = mouse[0]
            if keys[pygame.K_r]:
                self.start_reload()
            # defuse
            if self.bomb and keys[pygame.K_e] and math.hypot(self.bomb[0] - self.px, self.bomb[1] - self.py) < 1.4 \
                    and self.speed_now == 0:
                self.defuse += dt
                if self.defuse >= DEFUSE_TIME:
                    self.end_round(True, "BOMB DEFUSED - COUNTER-TERRORISTS WIN")
            else:
                self.defuse = max(0, self.defuse - dt * 2)

        self.cd = max(0, self.cd - dt)
        self.kick = max(0, self.kick - dt * 3)
        self.hitmark = max(0, self.hitmark - dt)
        self.hurt = max(0, self.hurt - dt)
        self.pitch += (0 - self.pitch) * min(1, dt * 1.5) if abs(self.pitch) > 0 and not mouse[0] else 0
        if self.reload_t > 0:
            self.reload_t -= dt
            if self.reload_t <= 0:
                need = w["mag"] - self.mags[self.weapon]
                take = min(need, self.reserve[self.weapon])
                self.mags[self.weapon] += take
                self.reserve[self.weapon] -= take
        for k in self.killfeed:
            k[1] -= dt
        self.killfeed = [k for k in self.killfeed if k[1] > 0]
        self.msg_t = max(0, self.msg_t - dt)

        # phases
        if self.phase == "buy":
            self.timer -= dt
            if self.timer <= 0:
                self.phase, self.timer = "live", ROUND_TIME
        elif self.phase == "live":
            self.timer -= dt
            self.update_bots(dt)
            if self.timer <= 0:
                self.end_round(True, "TIME UP - COUNTER-TERRORISTS WIN")
        elif self.phase == "bomb":
            x, y, t = self.bomb
            t -= dt
            self.bomb = (x, y, t)
            if int(t * 2) != int((t + dt) * 2) and t < 10:
                self.play("beep")
            self.update_bots(dt)
            if t <= 0:
                self.boom = 1.0
                self.play("boom")
                self.end_round(False, "BOMB EXPLODED - TERRORISTS WIN")
        elif self.phase == "end":
            self.boom = max(0, self.boom - dt)
            self.end_t -= dt
            if self.end_t <= 0:
                if self.ct_score >= WIN_ROUNDS or self.t_score >= WIN_ROUNDS:
                    self.state = "over"
                    pygame.event.set_grab(False)
                    pygame.mouse.set_visible(True)
                else:
                    keep_hp = self.hp > 0
                    self.start_round()
                    if not keep_hp:
                        pass

        if self.phase in ("live", "bomb") and self.hp <= 0:
            self.end_round(False, "YOU DIED - TERRORISTS WIN")
        if self.phase == "live" and all(b.hp <= 0 for b in self.bots):
            self.end_round(True, "ALL TERRORISTS ELIMINATED - CT WIN")

    # ---- render 3D
    def render(self):
        dx, dy = math.cos(self.ang), math.sin(self.ang)
        pl = PLANE * (0.35 if self.scoped else 1.0)
        plx, ply = -dy * pl, dx * pl
        perp, side, wallx, tile = cast_rays(self.px, self.py, dx, dy, plx, ply)
        self.zbuf = perp
        pitch = int(self.pitch)
        bob = math.sin(self.bob) * 2.0 if self.speed_now > 0 else 0
        horizon = RH / 2 + pitch + bob
        lh = ((RW / (2 * pl)) / perp).astype(np.float32)
        top = (horizon - lh / 2).astype(np.float32)
        bot = (horizon + lh / 2).astype(np.float32)

        off = MAXP - pitch - int(bob)
        off = int(np.clip(off, 0, 2 * MAXP))
        bg = BG[:, off:off + RH, :]
        ys = np.arange(RH, dtype=np.float32)[None, :]
        mask = (ys >= top[:, None]) & (ys < bot[:, None])
        row = (ys - top[:, None]) * (6.0 / lh[:, None])
        rfl = np.floor(row)
        brick = (wallx.astype(np.float32)[:, None] * 3 + (rfl % 2) * 0.5) % 1
        mortar = ((row - rfl) < 0.07) | (brick < 0.035)
        shade = (np.clip(1.5 / (1 + perp * 0.45), 0.12, 1.0) * np.where(side == 1, 0.72, 1.0)).astype(np.float32)
        base = WALL_COL[tile] * shade[:, None]
        tex = (1.0 - 0.38 * mortar).astype(np.float32)
        wall = base[:, None, :] * tex[..., None]
        frame = np.where(mask[..., None], wall, bg)
        pygame.surfarray.blit_array(self.small, frame.astype(np.uint8))

        # sprites
        sprites = []
        for b in self.bots:
            if b.hp > 0:
                sprites.append(("bot", b, b.x, b.y))
        if self.bomb:
            sprites.append(("bomb", None, self.bomb[0], self.bomb[1]))
        inv = 1.0 / (plx * dy - dx * ply)
        proj = []
        for kind, b, x, y in sprites:
            rx, ry = x - self.px, y - self.py
            tx = inv * (dy * rx - dx * ry)
            ty = inv * (-ply * rx + plx * ry)
            if ty > 0.15:
                proj.append((ty, tx, kind, b))
        proj.sort(key=lambda p: -p[0])
        for ty, tx, kind, b in proj:
            sx = RW / 2 * (1 + tx / ty)
            h = (RW / (2 * pl)) / ty
            floor_y = horizon + h / 2
            fog = float(np.clip(1.5 / (1 + ty * 0.45), 0.25, 1.0))
            if kind == "bot":
                sh = h * 0.9
                w2 = h * 0.2
                parts = [(0.0, 0.22, (222, 178, 140), 0.5), (0.22, 0.62, (176, 128, 70), 1.0),
                         (0.62, 1.0, (70, 64, 52), 0.8)]
                if b.flash > 0:
                    parts = [(a, c, (255, 255, 255), w) for a, c, _, w in parts]
            else:
                sh = h * 0.18
                w2 = h * 0.12
                on = int(self.bomb[2] * 2) % 2 == 0
                parts = [(0.0, 1.0, (255, 40, 30) if on else (120, 20, 20), 1.0)]
            tp = floor_y - sh
            for c in range(int(sx - w2), int(sx + w2) + 1):
                if 0 <= c < RW and ty < perp[c]:
                    for a, e, colr, wf in parts:
                        if abs(c - sx) <= w2 * wf:
                            y0, y1 = int(tp + sh * a), int(tp + sh * e)
                            if y1 > y0:
                                self.small.fill((int(colr[0] * fog), int(colr[1] * fog), int(colr[2] * fog)),
                                                (c, y0, 1, y1 - y0))

    # ---- HUD
    def text(self, surf, s, pos, font=None, col=(255, 255, 255), center=False, shadow=True):
        f = font or self.font
        img = f.render(s, True, col)
        r = img.get_rect()
        if center:
            r.center = pos
        else:
            r.topleft = pos
        if shadow:
            surf.blit(f.render(s, True, (0, 0, 0)), (r.x + 2, r.y + 2))
        surf.blit(img, r)

    def draw_hud(self):
        W, H = self.win.get_size()
        w = WEAPONS[self.weapon]
        # scope overlay / crosshair
        if self.scoped:
            pygame.draw.circle(self.win, (0, 0, 0), (W // 2, H // 2), int(H * 0.62), int(H * 0.28))
            pygame.draw.line(self.win, (0, 0, 0), (0, H // 2), (W, H // 2), 1)
            pygame.draw.line(self.win, (0, 0, 0), (W // 2, 0), (W // 2, H), 1)
        else:
            gap = 6 + int((w["spread"] * 300) * (1 + (2.2 if self.speed_now > 0 else 0))) + int(self.kick * 8)
            cx, cy = W // 2, H // 2
            for (x0, y0, x1, y1) in [(cx - gap - 10, cy, cx - gap, cy), (cx + gap, cy, cx + gap + 10, cy),
                                     (cx, cy - gap - 10, cx, cy - gap), (cx, cy + gap, cx, cy + gap + 10)]:
                pygame.draw.line(self.win, (0, 255, 120), (x0, y0), (x1, y1), 2)
        if self.hitmark > 0:
            cx, cy = W // 2, H // 2
            col = (255, 60, 60) if self.hitmark > 0.2 else (255, 255, 255)
            for sx_, sy_ in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                pygame.draw.line(self.win, col, (cx + sx_ * 6, cy + sy_ * 6), (cx + sx_ * 14, cy + sy_ * 14), 2)

        # view model
        if not self.scoped and self.hp > 0:
            bobx = math.sin(self.bob) * 10 if self.speed_now > 0 else 0
            boby = abs(math.cos(self.bob)) * 8 if self.speed_now > 0 else 0
            rel = 160 if self.reload_t > 0 else 0
            kx, ky = 0, self.kick * 28
            gx, gy = W * 0.62 + bobx, H * 0.78 + boby + ky + rel
            sizes = {"Pistol": (70, 120), "SMG": (90, 190), "Rifle": (100, 260), "Sniper": (90, 300)}
            gw, gl = sizes[self.weapon]
            pygame.draw.rect(self.win, (40, 40, 45), (gx - gw / 2, gy - gl * 0.2, gw, gl), border_radius=8)
            pygame.draw.rect(self.win, (70, 70, 78), (gx - gw / 2 + 8, gy - gl * 0.2, gw - 16, gl * 0.35), border_radius=6)
            pygame.draw.rect(self.win, (25, 25, 28), (gx - 14, gy - gl * 0.2 - 30, 28, 40))
            if self.kick > 0.6:
                pygame.draw.circle(self.win, (255, 220, 120), (int(gx), int(gy - gl * 0.2 - 40)), 22)
                pygame.draw.circle(self.win, (255, 255, 230), (int(gx), int(gy - gl * 0.2 - 40)), 10)

        # screen effects
        if self.hurt > 0:
            ov = pygame.Surface((W, H), pygame.SRCALPHA)
            ov.fill((200, 0, 0, int(140 * min(1, self.hurt / 0.35))))
            self.win.blit(ov, (0, 0))
        if self.boom > 0:
            ov = pygame.Surface((W, H), pygame.SRCALPHA)
            ov.fill((255, 200, 120, int(230 * self.boom)))
            self.win.blit(ov, (0, 0))

        # minimap
        cell = 7
        mm = pygame.Surface((MW * cell, MH * cell), pygame.SRCALPHA)
        mm.fill((0, 0, 0, 120))
        for y in range(MH):
            for x in range(MW):
                if GRID[y, x]:
                    pygame.draw.rect(mm, (110, 110, 120, 200), (x * cell, y * cell, cell, cell))
        for n, (sx_, sy_) in SITES.items():
            pygame.draw.rect(mm, (255, 200, 0, 160), (int(sx_ * cell) - 8, int(sy_ * cell) - 8, 16, 16), 1)
            mm.blit(self.tiny.render(n, True, (255, 200, 0)), (int(sx_ * cell) - 4, int(sy_ * cell) - 7))
        if self.bomb:
            pygame.draw.circle(mm, (255, 40, 40), (int(self.bomb[0] * cell), int(self.bomb[1] * cell)), 4)
        for b in self.bots:
            if b.hp > 0 and line_of_sight(self.px, self.py, b.x, b.y) and \
                    math.hypot(b.x - self.px, b.y - self.py) < 12:
                pygame.draw.circle(mm, (255, 80, 60), (int(b.x * cell), int(b.y * cell)), 3)
        pygame.draw.circle(mm, (80, 200, 255), (int(self.px * cell), int(self.py * cell)), 3)
        pygame.draw.line(mm, (80, 200, 255), (self.px * cell, self.py * cell),
                         (self.px * cell + math.cos(self.ang) * 12, self.py * cell + math.sin(self.ang) * 12), 2)
        self.win.blit(mm, (10, 10))

        # bottom bars
        hp_col = (255, 80, 80) if self.hp < 30 else (255, 255, 255)
        self.text(self.win, f"+ {max(0, int(self.hp))}", (24, H - 70), self.big, hp_col)
        self.text(self.win, f"ARMOR {int(self.armor)}", (28, H - 108), self.font, (120, 190, 255))
        ammo = f"{self.mags[self.weapon]} / {self.reserve[self.weapon]}"
        self.text(self.win, ammo, (W - 24 - self.big.size(ammo)[0], H - 70), self.big)
        self.text(self.win, self.weapon.upper() + ("  RELOADING" if self.reload_t > 0 else ""),
                  (W - 24 - self.font.size(self.weapon.upper() + ("  RELOADING" if self.reload_t > 0 else ""))[0], H - 104),
                  self.font, (255, 220, 120))
        self.text(self.win, f"$ {self.money}", (W - 160, 20), self.font, (120, 255, 120))

        # top center: score + timer
        mins = int(max(0, self.timer)) // 60
        secs = int(max(0, self.timer)) % 60
        label = f"{mins}:{secs:02d}"
        if self.phase == "bomb":
            label = f"BOMB {self.bomb[2]:4.1f}"
        elif self.phase == "buy":
            label = f"BUY {int(self.timer)}"
        self.text(self.win, f"CT {self.ct_score}   {label}   T {self.t_score}", (W // 2, 24), self.font, center=True)

        # killfeed
        for i, (s, _) in enumerate(self.killfeed[:5]):
            self.text(self.win, s, (W - 20 - self.tiny.size(s)[0], 50 + i * 20), self.tiny, (255, 255, 255))

        # buy menu
        if self.phase == "buy":
            self.text(self.win, "BUY PHASE  (press 1-4)", (W // 2, 90), self.font, (255, 220, 120), center=True)
            for i, name in enumerate(ORDER):
                own = name in self.owned
                afford = self.money >= WEAPONS[name]["price"]
                col = (120, 255, 120) if own else ((255, 255, 255) if afford else (130, 130, 130))
                tag = "OWNED" if own else f"${WEAPONS[name]['price']}"
                self.text(self.win, f"{i + 1}. {name:<7} {tag}", (W // 2 - 130, 125 + i * 28), self.font, col)

        # defuse bar
        if self.bomb:
            if self.defuse > 0:
                bw = 300
                pygame.draw.rect(self.win, (0, 0, 0), (W // 2 - bw // 2 - 2, H // 2 + 80, bw + 4, 24))
                pygame.draw.rect(self.win, (80, 200, 255), (W // 2 - bw // 2, H // 2 + 82, bw * self.defuse / DEFUSE_TIME, 20))
                self.text(self.win, "DEFUSING", (W // 2, H // 2 + 70), self.tiny, center=True)
            elif math.hypot(self.bomb[0] - self.px, self.bomb[1] - self.py) < 1.4:
                self.text(self.win, "HOLD E TO DEFUSE (stand still)", (W // 2, H // 2 + 90), self.font, center=True)
        carrier = next((b for b in self.bots if b.hp > 0 and b.carrier and b.plant > 0), None)
        if carrier and not self.bomb:
            self.text(self.win, "PLANTING...", (W // 2, 70), self.font, (255, 120, 80), center=True)
        if self.msg_t > 0:
            self.text(self.win, self.msg, (W // 2, H // 3), self.font, (255, 255, 120), center=True)
        if self.hp <= 0 and self.phase != "end":
            self.text(self.win, "YOU DIED", (W // 2, H // 2), self.big, (255, 60, 60), center=True)

    def menu(self, over=False):
        W, H = self.win.get_size()
        self.win.fill((14, 16, 22))
        self.text(self.win, "CS ARENA", (W // 2, H // 4), self.big, (255, 190, 80), center=True)
        if over:
            res = "YOU WIN THE MATCH!" if self.ct_score > self.t_score else "TERRORISTS WIN THE MATCH"
            self.text(self.win, f"{res}   {self.ct_score} : {self.t_score}", (W // 2, H // 4 + 70), self.font, center=True)
        self.text(self.win, "Bomb defusal vs bots. First to %d rounds." % WIN_ROUNDS, (W // 2, H // 2 - 40), self.font, center=True)
        for i, d in enumerate(DIFF):
            col = (120, 255, 120) if d == self.diff else (200, 200, 200)
            self.text(self.win, f"F{i + 1}  {d}", (W // 2, H // 2 + i * 34), self.font, col, center=True)
        self.text(self.win, "ENTER = start     ESC = quit", (W // 2, H // 2 + 140), self.font, center=True)
        help_lines = ["WASD move  |  Mouse look  |  LMB fire  |  RMB scope (sniper)  |  R reload",
                      "Shift walk  |  E hold to defuse  |  1-4 buy/switch  |  Q last weapon  |  [ ] sensitivity"]
        for i, l in enumerate(help_lines):
            self.text(self.win, l, (W // 2, H - 80 + i * 24), self.tiny, (170, 170, 180), center=True)

    # ---- main loop
    def run(self):
        self._mouse_was = False
        self.sprint = True
        self.speed_now = 0.0
        while self.running:
            dt = min(0.05, self.clock.tick(FPS_CAP) / 1000.0)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        if self.state == "play":
                            self.state = "menu"
                            pygame.event.set_grab(False)
                            pygame.mouse.set_visible(True)
                        else:
                            self.running = False
                    elif e.key == pygame.K_F11:
                        self.fullscreen = not self.fullscreen
                        self.win = pygame.display.set_mode(
                            (0, 0) if self.fullscreen else (WIN_W, WIN_H),
                            pygame.FULLSCREEN if self.fullscreen else pygame.RESIZABLE)
                    elif self.state in ("menu", "over"):
                        if e.key in (pygame.K_F1, pygame.K_F2, pygame.K_F3):
                            self.diff = list(DIFF)[e.key - pygame.K_F1]
                        elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                            self.new_match()
                            self.state = "play"
                    elif self.state == "play":
                        if e.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                            self.buy(ORDER[e.key - pygame.K_1])
                        elif e.key == pygame.K_q:
                            self.switch(self.last_weapon)
                        elif e.key == pygame.K_LEFTBRACKET:
                            self.sens = max(0.0005, self.sens - 0.0003)
                        elif e.key == pygame.K_RIGHTBRACKET:
                            self.sens = min(0.01, self.sens + 0.0003)
            if self.state == "play":
                self.update(dt)
                self.render()
                pygame.transform.scale(self.small, self.win.get_size(), self.win)
                self.draw_hud()
                self.text(self.win, f"{int(self.clock.get_fps())} FPS  sens {self.sens * 1000:.1f}",
                          (10, self.win.get_height() - 24), self.tiny, (200, 200, 200))
            else:
                self.menu(self.state == "over")
            pygame.display.flip()
        pygame.quit()


if __name__ == "__main__":
    Game().run()
    sys.exit()
