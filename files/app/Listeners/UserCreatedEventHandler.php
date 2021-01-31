<?php

namespace App\Listeners;

use App\Events\UserCreated;
use App\Models\User;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Support\Facades\Hash;

class UserCreatedEventHandler
{
    /**
     * Create the event listener.
     *
     * @return void
     */
    public function __construct()
    {
        //
    }

    /**
     * Handle the event.
     *
     * @param  UserCreated  $event
     * @return void
     */
    public function handle(UserCreated $event)
    {
        return User::create([
            'name' => $event['name'],
            'email' => $event['email'],
            'password' => $event
        ]);
    }
}
