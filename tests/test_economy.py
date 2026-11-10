import itertools
import math
import random
import unittest
from dataclasses import replace

from cs2_money_round.buy import FULL_BUY_TARGET, BuyTarget, armor_cost, estimate, evaluate
from cs2_money_round.model import Player, Rules, Team
from cs2_money_round.simulation import Action, Outcome, reset_half, settle


def team(cash=3350, side='T', **changes):
    return Team(side, tuple(Player(str(i), cash, **changes) for i in range(5)))


def actions(t, *, spent=0, died=True):
    return tuple(Action(p.id, spent, 0, died, None, 0) for p in t.players)


class BuyTests(unittest.TestCase):
    def test_exact_minimum_for_each_side(self):
        for side, minimum in [('T', 3350), ('CT', 3550)]:
            self.assertTrue(evaluate(team(minimum, side))['full_buy_possible'])
            self.assertFalse(evaluate(team(minimum-1, side))['full_buy_possible'])

    def test_rich_player_can_drop_four_guns(self):
        t = team(650)
        t = replace(t, players=(replace(t.players[0], cash=14150), *t.players[1:]))
        self.assertTrue(evaluate(t)['full_buy_possible'])

    def test_team_total_cannot_pay_someone_elses_armor(self):
        t = team(4000)
        t = replace(t, players=(replace(t.players[0], cash=649), *t.players[1:]))
        self.assertFalse(evaluate(t)['full_buy_possible'])
        self.assertEqual(evaluate(t)['max_equipped_players'], 4)

    def test_spare_cash_cannot_be_pooled(self):
        t = team(650, rifle='ak47')
        ps = [replace(p, cash=1190) for p in t.players]
        ps[0] = replace(ps[0], rifle=None)
        self.assertFalse(evaluate(replace(t, players=tuple(ps)))['full_buy_possible'])

    def test_saved_rifles_and_armor_need_no_cash(self):
        self.assertTrue(evaluate(team(0, rifle='m4a4', armor=20))['full_buy_possible'])
        self.assertFalse(evaluate(team(0, rifle='m4a4', armor=20), target=BuyTarget(min_armor=100))['full_buy_possible'])

    def test_saved_rifle_can_go_to_teammate(self):
        t = team(0, armor=100)
        ps = list(t.players)
        ps[0] = replace(ps[0], rifle='ak47', cash=2700)
        for i in range(2, 5):
            ps[i] = replace(ps[i], rifle='ak47')
        self.assertTrue(evaluate(replace(t, players=tuple(ps)))['full_buy_possible'])

    def test_project_full_buy_thresholds(self):
        for side, minimum in [('T', 3700), ('CT', 3900)]:
            self.assertTrue(evaluate(team(minimum, side), target=FULL_BUY_TARGET)['full_buy_possible'])
            self.assertFalse(evaluate(team(minimum - 1, side), target=FULL_BUY_TARGET)['full_buy_possible'])

    def test_project_full_buy_counts_damaged_armor_with_helmet(self):
        # Owner-Entscheid 10.11.2026: beschädigte Rüstung + Helm muss nicht auf 100 aufgefüllt werden.
        self.assertEqual(armor_cost(Player('p', 0, armor=1, helmet=True), FULL_BUY_TARGET, Rules()), 0)
        self.assertTrue(evaluate(team(2700, 'T', armor=35, helmet=True), target=FULL_BUY_TARGET)['full_buy_possible'])

    def test_helmet_upgrade_and_damaged_armor(self):
        target = BuyTarget(helmet=True)
        self.assertEqual(armor_cost(Player('p', 0, armor=100), target, Rules()), 350)
        self.assertEqual(armor_cost(Player('p', 0, armor=99), target, Rules()), 1000)
        self.assertEqual(armor_cost(Player('p', 0, armor=20, helmet=True), target, Rules()), 0)

    def test_reserve_is_personal(self):
        self.assertFalse(evaluate(team(3350), target=BuyTarget(reserve=1000))['full_buy_possible'])
        self.assertTrue(evaluate(team(4350), target=BuyTarget(reserve=1000))['full_buy_possible'])

    def test_solver_against_brute_force_purchase_assignments(self):
        rng = random.Random(44)
        rules = Rules(ak_price=3, m4_price=4, kevlar_price=2)
        for _ in range(100):
            ps = tuple(Player(str(i), rng.randrange(10), 'ak47' if rng.randrange(2) else None,
                              100 if rng.randrange(2) else 0) for i in range(5))
            t = Team('T', ps)
            saved = sum(p.rifle is not None for p in ps)
            possible = False
            for purchases in itertools.product(range(4), repeat=5):
                if saved + sum(purchases) < 5:
                    continue
                if all(p.cash >= 3*n + (0 if p.armor else 2) for p, n in zip(ps, purchases)):
                    possible = True
                    break
            self.assertEqual(evaluate(t, rules)['full_buy_possible'], possible)

    def test_weighted_scenarios_are_explicit(self):
        r = estimate([(3, team()), (1, team(0))])
        self.assertEqual(r['full_buy_probability'], .75)
        self.assertIn('not calibrated', r['probability_basis'])

    def test_invalid_weights_and_mixed_sides(self):
        for weight in [0, -1, math.inf, math.nan, True]:
            with self.assertRaises(ValueError):
                estimate([(weight, team())])
        with self.assertRaises(ValueError):
            estimate([(1, team()), (1, team(side='CT'))])

    def test_input_validation(self):
        for cash in [-1, True, 1.5]:
            with self.assertRaises(ValueError):
                Player('p', cash)
        with self.assertRaises(ValueError):
            evaluate(team(16001))
        with self.assertRaises(ValueError):
            Team('T', (Player('same', 0),)*5)


class SimulationTests(unittest.TestCase):
    def test_pistol_loss_and_next_bonus_are_distinct(self):
        t = team(800)
        next_t, ledger = settle(t, actions(t, spent=800), Outcome('CT', 'elimination', t_deaths=5))
        self.assertEqual(next_t.players[0].cash, 1900)
        self.assertEqual(ledger['players'][0]['loss_bonus_received'], 1900)
        self.assertEqual(ledger['loss_bonus_if_next_round_lost'], 2400)

    def test_loss_bonus_cap_then_win_then_loss(self):
        t = replace(team(0), loss_level=4)
        t, _ = settle(t, actions(t, died=False), Outcome('T', 'elimination'))
        self.assertEqual(t.loss_level, 3)
        t, ledger = settle(t, actions(t), Outcome('CT', 'elimination', t_deaths=5))
        self.assertEqual(ledger['players'][0]['loss_bonus_received'], 2900)
        self.assertEqual(t.loss_level, 4)

    def test_multiple_losses_saturate(self):
        t = team(0)
        for _ in range(10):
            t, _ = settle(t, actions(t), Outcome('CT', 'elimination', t_deaths=5))
        self.assertEqual(t.loss_level, 4)
        self.assertEqual(t.players[0].cash, 16000)

    def test_surviving_t_on_timeout_gets_no_loss_bonus(self):
        t = team(100)
        a = list(actions(t))
        a[0] = Action('0', 0, 300, False, 'ak47', 70)
        t, ledger = settle(t, tuple(a), Outcome('CT', 'timeout', t_deaths=4))
        self.assertEqual(t.players[0].cash, 400)
        self.assertEqual(t.players[0].rifle, 'ak47')
        self.assertEqual(t.players[1].cash, 2000)
        self.assertEqual(ledger['players'][0]['loss_bonus_received'], 0)

    def test_t_plant_and_defuse_loss(self):
        t = team(0)
        t, ledger = settle(t, actions(t), Outcome('CT', 'defuse', True, '0', 'enemy', 5))
        self.assertEqual(t.players[0].cash, 2800)
        self.assertEqual(t.players[1].cash, 2500)
        self.assertEqual(ledger['players'][0]['personal_objective_income'], 300)

    def test_ct_team_award_and_defuser(self):
        t = team(0, 'CT')
        a = actions(t, died=False)
        t, _ = settle(t, a, Outcome('CT', 'defuse', True, 'enemy', '0', 4))
        self.assertEqual(t.players[0].cash, 4000)
        self.assertEqual(t.players[1].cash, 3700)

    def test_spending_and_kill_rewards(self):
        t = team(4000)
        a = list(actions(t, spent=3350, died=False))
        a[0] = Action('0', 3350, 600, False, 'ak47', 100)
        t, _ = settle(t, tuple(a), Outcome('T', 'elimination'))
        self.assertEqual(t.players[0].cash, 4500)
        self.assertEqual(t.players[0].armor, 100)

    def test_money_cap_is_in_ledger(self):
        t = team(16000)
        t, ledger = settle(t, actions(t, died=False), Outcome('T', 'elimination'))
        self.assertEqual(t.players[0].cash, 16000)
        self.assertEqual(ledger['players'][0]['cap_discarded'], 3250)

    def test_explicit_half_reset(self):
        t = reset_half(team(15000, rifle='ak47', armor=100))
        self.assertEqual(t.side, 'CT')
        self.assertEqual(t.loss_level, 1)
        self.assertEqual(t.players[0], Player('0', 800))

    def test_reject_inconsistent_rounds(self):
        with self.assertRaises(ValueError):
            Outcome('T', 'timeout')
        with self.assertRaises(ValueError):
            Outcome('CT', 'elimination', bomb_planted=True)
        with self.assertRaises(ValueError):
            Action('0', 0, 0, True, 'ak47', 0)
        t = team(0)
        with self.assertRaises(ValueError):
            settle(t, actions(t, spent=1), Outcome('CT', 'elimination', t_deaths=5))
        with self.assertRaises(ValueError):
            settle(t, actions(t), Outcome('CT', 'elimination', t_deaths=4))
        with self.assertRaises(ValueError):
            settle(t, actions(t), Outcome('CT', 'defuse', bomb_planted=True, t_deaths=5))


if __name__ == '__main__':
    unittest.main()
