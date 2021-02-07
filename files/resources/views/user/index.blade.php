@extends('layouts.app')

@section('content')
    <div class="container">
        <table class="table table-striped">
            <thead>
            <tr>
                <th>name</th>
                <th>email</th>
            </tr>
            </thead>
            <tbody>

            @foreach($users as $user)
                <tr>
                    <td>{{$user->name}}</td>
                    <td>{{$user->email}}</td>
                    <td><a href="{{action('UserDocumentController@store', $user->id)}}" class="btn btn-warning">Upload document</a></td>
                    <td>
                        <form action="{{action('UserDocumentController@destroy', $user->id)}}" method="post">
                            @csrf
                            <input name="_method" type="hidden" value="DELETE">
                            <button class="btn btn-danger" type="submit">Delete document</button>
                        </form>
                    </td>
                </tr>
            @endforeach
            </tbody>
        </table>
        <div class="row justify-content-center">
            <div class="col-md-8">
                @foreach($users as $user)
                    <div class="card mt-4">
                        <div class="card-header">{{ $user->name }}</div>
                    </div>
                @endforeach
            </div>
        </div>
    </div>
    </div>
@endsection

