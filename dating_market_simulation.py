# =========================================================
# Dating Market Simulation v4.1 - Group C (Optimized Visual Demo)
# =========================================================

import pygame
import random
import math
import pandas as pd
from modelos_grupoA import ModelosGrupoA

# ------------------------------
# Clase Agent
# ------------------------------
class Agent:
    def __init__(self, x, y, gender, attr, fun, shar):
        self.x = x
        self.y = y
        self.gender = gender
        self.attr = attr
        self.fun = fun
        self.shar = shar
        self.color = (66, 135, 245) if gender == 'M' else (245, 99, 173)
        self.radius = 8
        self.vx = random.choice([-2, -1, 1, 2])
        self.vy = random.choice([-2, -1, 1, 2])
        self.matched = False

    def move(self, width, height):
        if self.matched: return
        self.x += self.vx
        self.y += self.vy
        if self.x <= 10 or self.x >= width - 10: self.vx *= -1
        if self.y <= 10 or self.y >= height - 10: self.vy *= -1

    def draw(self, screen):
        # halo
        pygame.draw.circle(screen, (255,255,255,40), (int(self.x), int(self.y)), self.radius+3)
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)


# ------------------------------
# Clase principal
# ------------------------------
class DatingMarketSimulationV41:
    def __init__(self, n_agents=50, width=1280, height=720, rules_path="apriori_rules_GroupA.csv"):
        pygame.init()
        self.width, self.height = width, height
        self.margin_right = 380
        self.world_width = width - self.margin_right
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("💘 Dating Market Simulation v4.1 - Enhanced Edition")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 22)

        # Modelo
        self.model = ModelosGrupoA("data/speed_dating_cleaned.csv")
        self.model.cargar_datos()
        self.model.entrenar_modelos()
        self.tree = self.model.models["Decision Tree"]

        # Reglas
        df = pd.read_csv(rules_path)
        df = df[df["antecedents"].str.contains("attr|fun|shar", case=False, na=False)]
        self.rules = [{"text": r["antecedents"], "strength": min(1.0, 0.6 + r["lift"]/2)} for _, r in df.iterrows()][:5]

        # Estado
        self.n_agents = n_agents
        self.diversity = 3
        self.agents = []
        self.total_matches = 0
        self.total_interactions = 0
        self.contact_memory = set()
        self.matches_log = []
        self.create_agents()

    # Crear agentes
    def create_agents(self):
        self.agents.clear()
        for i in range(self.n_agents):
            g = 'M' if i < self.n_agents / 2 else 'F'
            base = 5
            attr = max(1, min(10, base + random.randint(-self.diversity, self.diversity)))
            fun = max(1, min(10, base + random.randint(-self.diversity, self.diversity)))
            shar = max(1, min(10, base + random.randint(-self.diversity, self.diversity)))
            x, y = random.randint(50, self.world_width - 50), random.randint(50, self.height - 50)
            self.agents.append(Agent(x, y, g, attr, fun, shar))
        self.total_matches = 0
        self.total_interactions = 0
        self.contact_memory.clear()
        self.matches_log.clear()

    # Fondo degradado
    def draw_background(self):
        c1, c2 = (30, 10, 40), (90, 0, 80)
        for y in range(self.height):
            r = y / self.height
            color = [int(c1[i] * (1 - r) + c2[i] * r) for i in range(3)]
            pygame.draw.line(self.screen, color, (0, y), (self.world_width, y))

    # Panel lateral
    def draw_panel(self):
        x = self.world_width
        s = pygame.Surface((self.margin_right, self.height))
        s.set_alpha(230)
        s.fill((40, 30, 60))
        self.screen.blit(s, (x, 0))

        lines = [
            "Dating Market Simulation v4.1 💞",
            f"Agents: {self.n_agents}",
            f"Diversity: ±{self.diversity}",
            f"Interactions: {self.total_interactions}",
            f"Matches: {self.total_matches}",
            f"Success Rate: {self.total_matches/max(1,self.total_interactions)*100:.1f}%",
            "",
            "Apriori Rules:"
        ]
        y = 20
        for line in lines:
            t = self.font.render(line, True, (255,255,255))
            self.screen.blit(t, (x + 15, y)); y += 26
        for rule in self.rules:
            t = self.font.render(f"- {rule['text']}", True, (250,200,100))
            self.screen.blit(t, (x + 25, y)); y += 22

        # Botón de reinicio
        btn_rect = pygame.Rect(x + 100, self.height - 80, 160, 40)
        pygame.draw.rect(self.screen, (255,105,180), btn_rect, border_radius=10)
        txt = self.font.render("Restart ♻️", True, (0,0,0))
        self.screen.blit(txt, (x + 115, self.height - 70))
        return btn_rect

    # Evaluar interacciones
    def check_interaction(self, a1, a2):
        if a1.gender == a2.gender or a1.matched or a2.matched: return
        pair = tuple(sorted((id(a1), id(a2))))
        dist = math.hypot(a1.x - a2.x, a1.y - a2.y)
        if dist < 25:
            if pair not in self.contact_memory:
                self.contact_memory.add(pair)
                self.total_interactions += 1
                df = pd.DataFrame({'attr_o':[a1.attr],'fun_o':[a1.fun],'int_corr':[abs(a1.shar - a2.shar)/10]})
                prediction = self.tree.predict(df)[0]
                # Reglas Apriori + prob extra
                for rule in self.rules:
                    r = rule["text"].lower(); s = rule["strength"]
                    if "attr" in r and a1.attr>7 and a2.attr>7 and random.random()<s: prediction=1
                    if "fun" in r and a1.fun>7 and a2.fun>7 and random.random()<s: prediction=1
                    if "shar" in r and a1.shar>7 and a2.shar>7 and random.random()<s: prediction=1
                if random.random()<0.2: prediction=1  # base boost
                if prediction==1:
                    a1.matched=a2.matched=True
                    self.total_matches+=1
                    self.matches_log.append((a1,a2))

    def draw_matches(self):
        for a1,a2 in self.matches_log:
            pygame.draw.line(self.screen,(50,255,100),(int(a1.x),int(a1.y)),(int(a2.x),int(a2.y)),2)

    # Loop
    def run(self):
        running=True
        while running:
            self.draw_background()
            btn_rect=self.draw_panel()

            for e in pygame.event.get():
                if e.type==pygame.QUIT: running=False
                elif e.type==pygame.KEYDOWN:
                    if e.key==pygame.K_UP: self.n_agents=min(100,self.n_agents+2); self.create_agents()
                    elif e.key==pygame.K_DOWN: self.n_agents=max(10,self.n_agents-2); self.create_agents()
                    elif e.key==pygame.K_RIGHT: self.diversity=min(5,self.diversity+1); self.create_agents()
                    elif e.key==pygame.K_LEFT: self.diversity=max(1,self.diversity-1); self.create_agents()
                    elif e.key==pygame.K_q: running=False
                elif e.type==pygame.MOUSEBUTTONDOWN:
                    if btn_rect.collidepoint(e.pos): self.create_agents()

            for ag in self.agents:
                ag.move(self.world_width,self.height)
                ag.draw(self.screen)
            for i in range(len(self.agents)):
                for j in range(i+1,len(self.agents)):
                    self.check_interaction(self.agents[i],self.agents[j])
            self.draw_matches()
            pygame.display.flip()
            self.clock.tick(30)
        pygame.quit()


if __name__=="__main__":
    sim=DatingMarketSimulationV41()
    sim.run()




