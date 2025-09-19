$ie = New-Object -ComObject "InternetExplorer.Application"
$ie.Visible = $true
$ie.Navigate("https://www.google.com")


while ($ie.Busy) { Start-Sleep -Milliseconds 500 }


$doc = $ie.Document
$searchBox = $doc.getElementsByName("q") | Select-Object -First 1
$searchBox.value = "python"
$searchBox.form.submit()
