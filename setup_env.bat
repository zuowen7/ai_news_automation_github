@echo off
echo Setup Environment for AI News
set /p "PASSWORD=Enter password: "
setx AI_NEWS_EMAIL_PASSWORD "%PASSWORD%"
echo Done.
pause