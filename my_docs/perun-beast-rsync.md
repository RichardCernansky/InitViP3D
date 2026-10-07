# Pulling from perun to beast2 through a laptop VPN

Perun is only reachable from the laptop (VPN). Beast2 is reachable from the
laptop too, but beast2 and perun cannot talk directly. A reverse SSH tunnel
makes the laptop the relay without staging any data on its disk.

```
perun  <--VPN--  laptop  --SSH-->  beast2
         (login01.perun.tuke.sk)    (port 2222 opened here)
```

## 1. Load the key into the agent — on the laptop

```bash
ssh-add ~/.ssh/perun
ssh-add -l                 # must list the key, not "no identities"
```

macOS, to survive reboots:

```bash
ssh-add --apple-use-keychain ~/.ssh/perun
```

The agent is what beast2 borrows to authenticate against perun. Agent
forwarding forwards the *agent*, not the key file — an unloaded key is
invisible to it.

## 2. Open the tunnel — on the laptop

```bash
ssh -A -R 2222:login01.perun.tuke.sk:22 cernanskyr@beast2
```

- `-A` — forward the SSH agent so beast2 can authenticate to perun
- `-R 2222:login01.perun.tuke.sk:22` — open port 2222 **on beast2**, tunnelled
  back through the laptop to perun's port 22

The host in `-R` must be a real hostname or IP. `~/.ssh/config` aliases are
resolved by the SSH client only; the forward does a plain DNS lookup, so an
alias fails with `connect_to perun: unknown host`.

## 3. Verify the hop — on beast2

```bash
ssh -p 2222 riceke350@localhost
```

Should land on perun. If it does, rsync will work.

## 4. Pull — on beast2

```bash
rsync -avzP -e 'ssh -p 2222' \
  riceke350@localhost:/mnt/home/riceke350/LiViP3D/work_dirs/ \
  /home/cernanskyr/LiViP3D/work_dirs/
```

Steps 2–4 must be the same SSH session — port 2222 exists only while that
connection lives.

## Reading the command

| Part | Meaning |
|---|---|
| `riceke350@localhost` | account on **perun**; `localhost:2222` is the tunnel mouth on beast2 |
| `/mnt/home/riceke350/...` | path as it exists on **perun** |
| `/home/cernanskyr/...` | path as it exists on **beast2** |
| `-e 'ssh -p 2222'` | tell rsync to dial the tunnel instead of port 22 |

The laptop never appears in a `user@host` — it is only ever the initiator.

## Flags

- `-a` archive: recursive, preserves perms, times, symlinks
- `-v` verbose
- `-z` compress in transit
- `-P` = `--partial --progress`: live per-file progress, and keep partial files
  so an interrupted transfer resumes instead of restarting

Rerunning the same command is always safe. rsync compares size and mtime,
skips matching files, and sends only changed blocks for the rest.

## Useful variants

```bash
# dry run — change nothing, especially before --delete
rsync -avzn ...

# exact mirror, removes files absent from the source
rsync -avzP --delete ...

# skip intermediate checkpoints
rsync -avzP --exclude='epoch_*.pth' ...

# re-copy was triggered by mtime drift (dest built by scp/unzip)
rsync -avzP --size-only ...     # or -c to compare by checksum
```

## Failure modes seen

| Symptom | Cause |
|---|---|
| `Could not resolve hostname perun` | used the alias as the rsync target instead of `localhost` |
| `connect_to perun: unknown host` | alias used in the `-R` argument; needs the real FQDN |
| `Permission denied (publickey)` | agent empty on the laptop, or connected before `ssh-add` |
| `kex_exchange_identification: Connection reset` | tunnel endpoint unreachable — check the VPN is up |

## Long transfers

Run rsync inside `tmux` on beast2. If the laptop sleeps or the VPN drops, the
tunnel dies but the rsync process survives; reopen the tunnel (step 2) and
rerun the same command, which resumes.

If perun turns out to need a `ProxyJump`, the reverse tunnel cannot work — a
port forward only makes a direct TCP connection. Stage through the laptop
instead, in two resumable legs:

```bash
rsync -avzP riceke350@perun:/mnt/home/riceke350/LiViP3D/work_dirs/ ~/staging/
rsync -avzP ~/staging/ cernanskyr@beast2:/home/cernanskyr/LiViP3D/work_dirs/
```