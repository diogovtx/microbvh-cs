# Downloads tinybvh's test scenes used by microBVH's tests, pinned to the commit of tinybvh's 1.8.0 release.
$base = "https://raw.githubusercontent.com/jbikker/tinybvh/0e4584287823252cf83f0e9cd072848bec5f79c5/testdata"
$dest = $PSScriptRoot
foreach ($name in @("bunny.bin", "suzanne.bin", "cryteksponza.bin"))
{
	$out = Join-Path $dest $name
	if (-not (Test-Path $out))
	{
		Write-Host "Fetching $name"
		Invoke-WebRequest -Uri "$base/$name" -OutFile $out
	}
}
