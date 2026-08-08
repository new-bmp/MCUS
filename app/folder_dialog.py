from __future__ import annotations

import os
import subprocess
from pathlib import Path


def choose_folder(initial_directory: str | None = None) -> str | None:
    """Open a reliable always-on-top Windows folder browser."""
    if os.name != "nt":
        raise RuntimeError("Native folder selection is currently supported on Windows only")

    initial = str(Path(initial_directory).expanduser().resolve()) if initial_directory else ""
    escaped_initial = initial.replace("'", "''")
    script = f"""
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;

public static class VLATopmostWindow
{{
    private static readonly IntPtr HWND_TOPMOST = new IntPtr(-1);
    private const UInt32 SWP_NOSIZE = 0x0001;
    private const UInt32 SWP_NOMOVE = 0x0002;
    private const UInt32 SWP_SHOWWINDOW = 0x0040;

    [DllImport("user32.dll")]
    private static extern bool SetWindowPos(
        IntPtr hWnd, IntPtr hWndInsertAfter, int X, int Y, int cx, int cy, UInt32 flags);

    [DllImport("user32.dll")]
    private static extern bool SetForegroundWindow(IntPtr hWnd);

    public static void ShowAboveAll(IntPtr window)
    {{
        SetWindowPos(window, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW);
        SetForegroundWindow(window);
    }}
}}
'@

$form = New-Object System.Windows.Forms.Form
$form.Text = 'alice blue - Select Folder'
$form.Width = 720
$form.Height = 540
$form.MinimumSize = New-Object System.Drawing.Size(520, 380)
$form.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen
$form.ShowInTaskbar = $true
$form.TopMost = $true
$form.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::Sizable
$form.Font = New-Object System.Drawing.Font('Segoe UI', 9)

$header = New-Object System.Windows.Forms.Panel
$header.Dock = [System.Windows.Forms.DockStyle]::Top
$header.Height = 70
$header.Padding = New-Object System.Windows.Forms.Padding(12, 10, 12, 8)

$label = New-Object System.Windows.Forms.Label
$label.Text = 'Folder path'
$label.Dock = [System.Windows.Forms.DockStyle]::Top
$label.Height = 22

$pathBox = New-Object System.Windows.Forms.TextBox
$pathBox.Dock = [System.Windows.Forms.DockStyle]::Bottom
$pathBox.Height = 28
$header.Controls.Add($pathBox)
$header.Controls.Add($label)

$tree = New-Object System.Windows.Forms.TreeView
$tree.Dock = [System.Windows.Forms.DockStyle]::Fill
$tree.HideSelection = $false
$tree.ShowLines = $true
$tree.ShowRootLines = $true
$tree.FullRowSelect = $true
$tree.BorderStyle = [System.Windows.Forms.BorderStyle]::FixedSingle

$footer = New-Object System.Windows.Forms.Panel
$footer.Dock = [System.Windows.Forms.DockStyle]::Bottom
$footer.Height = 54
$footer.Padding = New-Object System.Windows.Forms.Padding(12, 10, 12, 10)

$cancel = New-Object System.Windows.Forms.Button
$cancel.Text = 'Cancel'
$cancel.Width = 92
$cancel.Height = 30
$cancel.Dock = [System.Windows.Forms.DockStyle]::Right
$cancel.DialogResult = [System.Windows.Forms.DialogResult]::Cancel

$select = New-Object System.Windows.Forms.Button
$select.Text = 'Select Folder'
$select.Width = 112
$select.Height = 30
$select.Dock = [System.Windows.Forms.DockStyle]::Right
$select.Margin = New-Object System.Windows.Forms.Padding(0, 0, 8, 0)

$footer.Controls.Add($cancel)
$footer.Controls.Add($select)
$form.Controls.Add($tree)
$form.Controls.Add($footer)
$form.Controls.Add($header)
$form.AcceptButton = $select
$form.CancelButton = $cancel

function Add-VLAFolderNode([System.Windows.Forms.TreeNodeCollection]$collection, [string]$path, [string]$labelText) {{
    $node = New-Object System.Windows.Forms.TreeNode
    $node.Text = $labelText
    $node.Tag = $path
    # Never probe a directory merely to decide whether it has children.  A
    # network/removable drive can block that call and make the whole dialog
    # appear frozen during startup.  The real enumeration is deferred until
    # the user expands the node.
    $placeholder = New-Object System.Windows.Forms.TreeNode
    $placeholder.Text = 'Loading...'
    $placeholder.Tag = $null
    [void]$node.Nodes.Add($placeholder)
    [void]$collection.Add($node)
    return $node
}}

$desktop = [Environment]::GetFolderPath([Environment+SpecialFolder]::Desktop)
$documents = [Environment]::GetFolderPath([Environment+SpecialFolder]::MyDocuments)
if ($desktop -and (Test-Path -LiteralPath $desktop)) {{ [void](Add-VLAFolderNode $tree.Nodes $desktop 'Desktop') }}
if ($documents -and (Test-Path -LiteralPath $documents) -and $documents -ne $desktop) {{ [void](Add-VLAFolderNode $tree.Nodes $documents 'Documents') }}
[void](Add-VLAFolderNode $tree.Nodes '__VLA_DRIVES__' 'This PC')

$tree.Add_BeforeExpand({{
    param($sender, $eventArgs)
    $node = $eventArgs.Node
    if ($node.Nodes.Count -ne 1 -or $null -ne $node.Nodes[0].Tag) {{ return }}
    $node.Nodes.Clear()
    try {{
        if ([string]$node.Tag -eq '__VLA_DRIVES__') {{
            foreach ($drive in [System.IO.DriveInfo]::GetDrives()) {{
                if (-not $drive.IsReady) {{ continue }}
                $driveLabel = if ($drive.VolumeLabel) {{ "$($drive.Name)  $($drive.VolumeLabel)" }} else {{ $drive.Name }}
                [void](Add-VLAFolderNode $node.Nodes $drive.RootDirectory.FullName $driveLabel)
            }}
            return
        }}
        $directories = [System.IO.Directory]::EnumerateDirectories([string]$node.Tag) | Sort-Object
        foreach ($directory in $directories) {{
            $info = New-Object System.IO.DirectoryInfo($directory)
            if (($info.Attributes -band [System.IO.FileAttributes]::Hidden) -ne 0) {{ continue }}
            [void](Add-VLAFolderNode $node.Nodes $info.FullName $info.Name)
        }}
    }} catch {{}}
}})

$tree.Add_AfterSelect({{
    param($sender, $eventArgs)
    if ($eventArgs.Node.Tag -and [string]$eventArgs.Node.Tag -ne '__VLA_DRIVES__') {{ $pathBox.Text = [string]$eventArgs.Node.Tag }}
}})

$script:selectedPath = $null
$select.Add_Click({{
    $candidate = $pathBox.Text.Trim()
    if (Test-Path -LiteralPath $candidate -PathType Container) {{
        $script:selectedPath = [System.IO.Path]::GetFullPath($candidate)
        $form.DialogResult = [System.Windows.Forms.DialogResult]::OK
        $form.Close()
    }} else {{
        [System.Windows.Forms.MessageBox]::Show(
            $form,
            'The selected folder does not exist.',
            'alice blue',
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Warning
        ) | Out-Null
    }}
}})

$tree.Add_NodeMouseDoubleClick({{
    param($sender, $eventArgs)
    if ($eventArgs.Node.Tag) {{
        $pathBox.Text = [string]$eventArgs.Node.Tag
        $select.PerformClick()
    }}
}})

$startingPath = '{escaped_initial}'
if (-not $startingPath -or -not (Test-Path -LiteralPath $startingPath -PathType Container)) {{
    $startingPath = $desktop
}}
$pathBox.Text = $startingPath

$form.Add_Shown({{
    $form.BringToFront()
    $form.Activate()
    [VLATopmostWindow]::ShowAboveAll($form.Handle)
    $pathBox.Focus()
    $pathBox.SelectAll()
}})

$result = $form.ShowDialog()
if ($result -eq [System.Windows.Forms.DialogResult]::OK -and $script:selectedPath) {{
    [Console]::Write($script:selectedPath)
}}
$form.Dispose()
"""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    completed = subprocess.run(
        ["powershell.exe", "-NoLogo", "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=3600,
        creationflags=flags,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or "Windows folder dialog failed"
        raise RuntimeError(message)
    selected = completed.stdout.strip()
    if not selected:
        return None
    path = Path(selected).resolve()
    if not path.is_dir():
        raise RuntimeError(f"Selected folder does not exist: {path}")
    return str(path)
