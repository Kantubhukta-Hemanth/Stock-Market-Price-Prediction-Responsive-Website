from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static


from .import views

urlpatterns = [
    path('', views.index, name='index'),
    path('market/', views.market, name='market'),
    path('market/update', views.update, name='update'),
    path('market/info', views.info, name='info'),
    path('market/live', views.live, name='live'),
    path('market/predict', views.predict, name='predict'),
    path('update/', views.update, name='update'),
    path('register', views.register, name='register'),
    path('login', views.login, name='login'),
    path('logout', views.logout, name='logout'),
    path('profile', views.profile, name='profile'),
    path('market/buy', views.buy, name='buy'),
    path('market/sell', views.sell, name='sell')

]


#username    time    stock    quantity    buy_stockprice  sell_stockprice   buy_amount    status(buy/sell)    profit/loss