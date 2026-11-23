---
title: Fleet Maintenance Handbook (synthetic)
source_url: knowledge/fleet_handbook.md
licence: CC BY 4.0, written for this project
synthetic: true
---

# About this handbook

This handbook is SYNTHETIC. It was written for the Maintenance Copilot portfolio project
to match the simulated Microsoft predictive-maintenance sample fleet. The machines, part
names and procedures are invented. The operating ranges were derived from the simulated
telemetry. Do not use it for real equipment.

The handbook covers the fleet's machine models, its four main components, the sensor
signals and their normal ranges, the five error codes, troubleshooting steps, scheduled
maintenance, failure-risk alerts, ticket priorities and safety before work.

# Fleet overview and machine models

The fleet consists of identical-purpose production machines in four model variants:
model1, model2, model3 and model4. All models share the same four main components
(comp1 to comp4) and the same four sensors. Differences between the models are in the
generation of the control electronics. Older machines (age 15 years or more) need closer
attention, because worn mounts and connectors make faults more frequent.

Each machine reports one averaged sensor reading per hour. Error events are logged with
the hour in which they occurred. Every component replacement is recorded in the
maintenance log, either as a scheduled replacement or as a replacement after a failure.

# Components

## comp1 - Drive motor and power supply

comp1 is the drive motor with its power supply unit. It is monitored through the volt
signal (supply voltage). Typical failure signs are a rising supply voltage over several
hours and error1 events. A failed comp1 stops the machine.

## comp2 - Spindle and rotor assembly

comp2 is the spindle and rotor assembly. It is monitored through the rotate signal
(rotation speed). Typical failure signs are a falling rotation speed and error2 and
error3 events, which often appear together. comp2 is the component that fails most
often in this fleet.

## comp3 - Hydraulic pump and pressure system

comp3 is the hydraulic pump with its pressure lines and valves. It is monitored through
the pressure signal. Typical failure signs are a rising pressure and error4 events.

## comp4 - Bearings and mechanical mounts

comp4 covers the main bearings and the mechanical mounts. It is monitored through the
vibration signal. Typical failure signs are rising vibration and error5 events.

# Sensor signals and normal operating ranges

The ranges below come from the fleet's hourly averages. "Normal" is the band in which
90% of all readings lie. An hourly reading outside the alarm limit is unusual (about 1
in 100 readings) and should be checked against the error log.

| Signal | Unit | Normal band | Typical value | Alarm limit |
|---|---|---|---|---|
| volt | V | 146 - 197 | 171 | above 208 or below 135 |
| rotate | rpm | 359 - 531 | 448 | below 317 or above 566 |
| pressure | psi | 84 - 119 | 100 | above 131 or below 77 |
| vibration | mm/s | 32 - 49 | 40 | above 54 |

Single readings outside the normal band are common and not a fault by themselves. A
fault is likely when the 24-hour average moves away from the typical value, for example
a 24-hour average voltage above 185 V, rotation below 400 rpm, pressure above 115 psi or
vibration above 46 mm/s.

# Error codes

The machine controller raises five error codes. Errors do not stop the machine, but they
are early warnings. In this fleet, the error that belongs to a component usually appears
within the 24 hours before that component fails.

## error1 - Supply voltage out of tolerance

Related component: comp1 (drive motor and power supply).
Meaning: the controller measured a supply voltage outside its tolerance.
First checks: compare the volt signal of the last 24 hours with the normal band; check
the power supply connectors and the motor terminal box for heat damage.

## error2 - Rotor speed deviation

Related component: comp2 (spindle and rotor assembly).
Meaning: the rotor speed differs from the set point for more than one minute.
First checks: compare the rotate signal with the normal band; check the drive belt and
the spindle for drag or noise.

## error3 - Rotor torque limit

Related component: comp2 (spindle and rotor assembly).
Meaning: the rotor needed more torque than allowed. error3 usually appears together
with error2.
First checks: as for error2; also check the lubrication of the spindle.

## error4 - Hydraulic pressure high

Related component: comp3 (hydraulic pump and pressure system).
Meaning: the hydraulic pressure exceeded its limit.
First checks: compare the pressure signal with the normal band; check the pressure
relief valve, the filter and the lines for blockages.

## error5 - Vibration limit exceeded

Related component: comp4 (bearings and mechanical mounts).
Meaning: the vibration sensor exceeded its limit.
First checks: compare the vibration signal with the normal band; check the mounting
bolts and listen to the main bearings.

# Troubleshooting by symptom

## Repeated errors on one machine

If a machine raises the same error code two or more times within 24 hours, treat the
related component as likely to fail soon. Check the failure-risk model, inspect the
related component during the current shift and open a ticket with high priority.

## Several different error codes at once

error2 together with error3 points to comp2. Other combinations usually mean that more
than one component is worn. Start with the component whose error appeared most often in
the last 24 hours.

## Sensor drift without errors

A slow drift of a 24-hour average over several days, without error codes, is an early
sign of wear. Plan an inspection at the next scheduled maintenance visit and watch the
7-day failure risk.

## Machine just repaired

After a component replacement, readings can be unstable for a few hours. Errors in the
first 6 hours after a replacement should be logged but do not need a new ticket unless
they repeat.

# Scheduled maintenance

Maintenance visits take place on the 06:00 shift. Every machine is visited at least every
15 days. During a visit the technician inspects all four components and replaces worn
parts. Typical intervals between replacements of the same component on a machine are:
comp1 about 45 days, comp2 about 60 days, comp3 about 30 days and comp4 about 45 days.
A component that has not been replaced for much longer than its typical interval should
be inspected with priority at the next visit.

Every replacement is recorded in the maintenance log with the time, machine and
component. A replacement after a failure is also recorded as a failure.

# Failure-risk alerts

The fleet uses a machine-learning model that estimates, from the last 24 hours of
sensor readings, error counts, time since the last replacements, machine model and age,
the probability that a component fails within the next 24 hours and within the next 7
days. Each horizon has an alert threshold.

- 24-hour alert: inspect the machine in the current shift. Prepare the spare part for
  the most likely component. Open a ticket with high priority.
- 7-day alert (without 24-hour alert): plan an inspection of the most likely component
  at the next scheduled visit. Open a ticket with medium priority.
- No alert: no action beyond scheduled maintenance.

The model is a decision aid. A technician's observation (noise, smell, leaks) always
overrides it.

# Ticket priorities

Choose the priority of a maintenance ticket as follows:

- critical: a safety hazard (smoke, fire, leak of hydraulic fluid, exposed live parts) or
  a machine that has stopped. Stop the machine and inform the shift supervisor at once.
- high: the machine is on a 24-hour failure-risk alert, or the same error code appeared
  two or more times in the last 24 hours. Act within the current shift.
- medium: the machine is on a 7-day alert, or there are single errors or sensor drift.
  Handle at the next scheduled visit.
- low: routine observations, cleaning, labels, documentation.

A good ticket names the machine, the affected component if known, the priority, and a
short summary of the evidence (errors, readings, model risk).

# Safety before maintenance work

Before any service or maintenance on a machine, isolate all energy sources and apply
lockout/tagout according to the site procedure (see the OSHA 3120 guide on the control
of hazardous energy). Wait until moving parts have stopped and hydraulic pressure in
comp3 has been released before opening the pressure system. Only trained and authorised
technicians may remove their own lock. Never bypass machine guards; see the OSHA 3170
guide on safeguarding equipment.

# Documentation and escalation

Record every inspection and replacement in the maintenance log on the same shift. If a
machine fails twice within 30 days, the shift supervisor reviews the case with the
maintenance engineer. Questions about this handbook go to the maintenance engineering
team.
